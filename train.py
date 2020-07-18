# train file for bert2bert
import os
import logging
logging.basicConfig(level=logging.ERROR)
import time
import json
import shutil
import argparse
import numpy as np
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.autograd import Variable

from transformers import BertModel, BertConfig, BertTokenizer
from tok import Tok
from transformers import EncoderDecoderModel, EncoderDecoderConfig
from transformers import CONFIG_NAME
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from dataset_prep import SelectionDataset
from seq_transform import  SelectionSequentialTransform, SelectionJoinTransform
from encoder import BertPolyModel
from torch.nn import CrossEntropyLoss

def set_seed(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  # if args.n_gpu > 0:
  #   torch.cuda.manual_seed_all(args.seed)


def eval_running_model(dataloader):
  loss_fct = CrossEntropyLoss()
  model.eval()
  eval_loss, eval_hit_times, recall = 0, 0, 0
  nb_eval_steps, nb_eval_examples = 0, 0
  for step, batch in enumerate(dataloader, start=1):
    batch = tuple(t.to(device) for t in batch)
    context_ids_batch, context_segment_ids_list_batch, context_input_masks_list_batch, \
    response_input_ids, response_segment_ids_list_batch, response_input_masks_list_batch, _ = batch

    pos_resp_input_ids = response_input_ids[:, 0, :] # pick first one only, the true response BxS
    labels_batch = pos_resp_input_ids[:,1:].clone()
    labels_batch[labels_batch==0] = -100

    with torch.no_grad():
      loss, outputs = model(input_ids=context_ids_batch, decoder_input_ids=pos_resp_input_ids[:,:-1], labels=labels_batch)[:2]

    eval_hit_times += (outputs.argmax(-1)[:,:20] == labels_batch[:,:20]).sum().item()
    eval_loss += loss.item()

    nb_eval_examples += labels_batch.size(0)*labels_batch.size(1)
    nb_eval_steps += 1
  eval_loss = eval_loss / nb_eval_steps
  eval_accuracy = eval_hit_times / nb_eval_examples
  result = {
    'train_loss': tr_loss / nb_tr_steps,
    'eval_loss': eval_loss,
    'eval_accuracy': eval_accuracy,
    'epoch': epoch,
    'global_step': global_step,
    'labels': tokenizer.decode(labels_batch[0]),
    'output': tokenizer.decode(outputs.argmax(-1)[0])
  }
  return result

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='mean')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target, alpha):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx, as_tuple=False)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return alpha*self.criterion(x, Variable(true_dist, requires_grad=False))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_type", default='bert', type=str, help="Choose from bert or distilbert")
  parser.add_argument("--output_dir", required=True, type=str)
  parser.add_argument("--train_dir", default='./data/bst', type=str)

  parser.add_argument("--use_pretrain", action="store_true")

  parser.add_argument("--max_contexts_length", default=128, type=int)
  parser.add_argument("--max_response_length", default=128, type=int)
  parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
  parser.add_argument("--eval_batch_size", default=10, type=int, help="Total batch size for eval.")
  parser.add_argument("--print_freq", default=100, type=int, help="Prints every n iterations")

  # parser.add_argument("--poly_m", default=16, type=int, help="M query codes for poly-encoder, trainable")
  parser.add_argument("--max_history", default=30, type=int, help="max history")

  parser.add_argument("--learning_rate", default=5e-5, type=float, help="Initial learning rate for Adam.")
  parser.add_argument("--weight_decay", default=0.0, type=float)
  parser.add_argument("--warmup_steps", default=2000, type=float)
  parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam.")
  parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
  parser.add_argument("--alpha", default=0.3, type=float, help="Mixing hyperparam for ULE")

  parser.add_argument("--num_train_epochs", default=3.0, type=float)
  parser.add_argument('--seed', type=int, default=11111, help="random seed for initialization")
  parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                      help="Number of updates steps to accumulate before performing a backward/update pass.")
  parser.add_argument(
    "--fp16", # something new that i learnt
    action="store_true",
    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
  )
  parser.add_argument(
    "--fp16_opt_level",
    type=str,
    default="O1",
    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
         "See details at https://nvidia.github.io/apex/amp.html",
  )
  parser.add_argument('--gpu', type=int, default=0)
  args = parser.parse_args()
  print(args)
  os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu
  set_seed(args)

  MODEL_CLASSES = {
    'bert': (BertConfig, BertTokenizer, BertModel),
    'distilbert': (DistilBertConfig, DistilBertTokenizer, DistilBertModel)
  }
  ConfigClass, TokenizerClass, BertModelClass = MODEL_CLASSES[args.model_type]

  # init dataset and bert model
  tokenizerPE = TokenizerClass.from_pretrained(args.model_type+'-base-uncased')
  tokenizerGE = Tok(train_corpus="", save_file="")
  context_transform = SelectionJoinTransform(tokenizer=[tokenizerPE, tokenizerGE], max_len=args.max_contexts_length,
                                             max_history=args.max_history)
  response_transform = SelectionSequentialTransform(tokenizer=[tokenizerPE, tokenizerGE], max_len=args.max_response_length,
                                                    max_history=None, pair_last=False)

  print('=' * 80)
  print('Train dir:', args.train_dir)
  print('Output dir:', args.output_dir)
  print('=' * 80)

  train_dataset = SelectionDataset(os.path.join(args.train_dir, 'train_self_original.txt'),
                                   context_transform, response_transform, sample_cnt=None)
  val_dataset = SelectionDataset(os.path.join(args.train_dir, 'valid_self_original.txt'),
                                 context_transform, response_transform, sample_cnt=5000)
  train_dataloader = DataLoader(train_dataset,
                                batch_size=args.train_batch_size, collate_fn=train_dataset.batchify_join_str,
                                shuffle=True)
  val_dataloader = DataLoader(val_dataset,
                              batch_size=args.eval_batch_size, collate_fn=val_dataset.batchify_join_str, shuffle=False)
  t_total = len(train_dataloader) // args.train_batch_size * (max(5, args.num_train_epochs))

  epoch_start = 1
  global_step = 0
  best_eval_loss = float('inf')
  best_test_loss = float('inf')

  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

  log_wf = open(os.path.join(args.output_dir, 'log.txt'), 'a', encoding='utf-8') # for logging

  state_save_path = os.path.join(args.output_dir, 'pytorch_model.bin')
  output_dir = args.output_dir
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(device)

  ################################################################################
  # BERT encoder2decoder
  bert_config_e = ConfigClass(num_hidden_layers=8,hidden_size=512,num_attention_heads=16,intermediate_size=2048)
  bert_config_d = ConfigClass(num_hidden_layers=8,hidden_size=512,num_attention_heads=16,intermediate_size=2048)
  if args.use_pretrain:
    print('Loading parameters from hugging face %s-base-uncased'%args.model_type)
    log_wf.write('Loading parameters from hugging face %s-base-uncased \n'%args.model_type)
    model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased','bert-base-uncased')
  else:
    config = EncoderDecoderConfig.from_encoder_decoder_configs(bert_config_e, bert_config_d)
    model = EncoderDecoderModel(config)

  if os.path.exists(state_save_path):
      print ("Found pre-trained seq2seq checkpoint, recovering from there")
      model.load_state_dict(torch.load(state_save_path))
  model.to(device)

  no_decay = ["bias", "LayerNorm.weight"]
  optimizer_grouped_parameters = [
    {
      "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
      "weight_decay": args.weight_decay,
    },
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
  ]
  optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
  scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
  )
  if args.fp16:
    try:
      from apex import amp
    except ImportError:
      raise ImportError("install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

  tr_total = int(
    train_dataset.__len__() / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
  print_freq = args.print_freq
  eval_freq = min(len(train_dataloader) // 2, 1000)
  print('Print freq:', print_freq, "Eval freq:", eval_freq)
  crit = LabelSmoothing(tokenizer.vocab_size, 0, 0.2) # for unlikelihood loss

  path_for_PE = './out/pytorch_model_PE.bin'
  bert_config = ConfigClass()
  bert = BertModelClass(bert_config)
  PEmodel = BertPolyModel(bert_config, bert=bert, poly_m=64)

  if os.path.exists(state_save_path):
      print ("Found pre-trained poly-encoder checkpoint, recovering from there")
      PEmodel.load_state_dict(torch.load(path_for_PE, map_location=torch.device(device)))
  PEmodel.to(device)
  PEmodel.eval()

  for epoch in range(epoch_start, int(args.num_train_epochs) + 1):
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    with tqdm(total=len(train_dataloader)) as bar:
      for step, batch in enumerate(train_dataloader, start=1):
        model.train()
        optimizer.zero_grad()
        batch = tuple(t.to(device) for t in batch)
        context_ids_batchPE, context_segment_ids_list_batchPE, context_input_masks_list_batchPE, \
        context_ids_batchGE, context_segment_ids_list_batchGE, context_input_masks_list_batchGE, \
        response_input_idsPE, response_segment_ids_list_batchPE, response_input_masks_list_batchPE, \
        response_input_idsGE, response_segment_ids_list_batchGE, response_input_masks_list_batchGE, _ = batch

        context_ids_batch, context_segment_ids_list_batch, context_input_masks_list_batch, \
        response_input_ids, response_segment_ids_list_batch, response_input_masks_list_batch, _ = batch

        PEloss, picked_seq = PEmodel(context_token_ids_batchPE, context_segment_ids_list_batchPE, context_input_masks_list_batchPE,
                                response_input_idsPE, response_segment_ids_list_batchPE, response_input_masks_list_batchPE,
                                labels_batch) # Bx1 and BxS

        pos_resp_input_ids = response_input_ids[:, 0, :].clone() # pick first one only, the true response BxS
        labels_batch = pos_resp_input_ids[:,1:].clone()
        labels_batch[labels_batch==0] = -100 # replace pad with -100, to not compute loss for pad_ids

        # set responses of other entries in a batch as negative candidates
        neg_resp_input_ids = []
        for _ in pos_rep_input_ids[:]:
            neg_resp_input_ids.append( pos_rep_input_ids[pos_rep_input_ids!=_].view(-1,pos_rep_input_ids.shape(-1) )[:10] ) # 10 is the number of negative cands picked from other entries in a batch; considered for ULE
        neg_resp_input_ids = torch.stack(neg_resp_input_ids)
        neg_labels_batch = neg_resp_input_ids[:,:,1:].clone()

        #concat picked_seq by PE with context_input_GE
        

        mle_loss, outputs = model(input_ids=context_ids_batch, decoder_input_ids=pos_resp_input_ids[:,:-1], labels=labels_batch)[:2]# outputs is BxSxVocabsize
        losses=[]
        losses.append(mle_loss)
        for _ in range(neg_labels_batch.shape):
            losses.append( crit( outputs.contiguous().view(-1,outputs.size(-1)), neg_labels_batch[:,_,:].contiguous().view(-1), (-1)*args.alpha )/(neg_labels_batch.shape )
        loss = sum(losses)
        tr_loss += loss.item()
        nb_tr_examples += context_ids_batch.size(0)
        nb_tr_steps += 1

        if args.fp16:
          with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
          torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        else:
          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()
        if global_step < args.warmup_steps:
          scheduler.step()
        model.zero_grad()
        optimizer.zero_grad()
        global_step += 1

        if step % print_freq == 0:
          bar.update(min(print_freq, step))
          time.sleep(0.02)
          print(global_step, tr_loss / nb_tr_steps)
          log_wf.write('%d\t%f\n' % (global_step, tr_loss / nb_tr_steps))

        if global_step % eval_freq == 0:
          if global_step == 4000:
            eval_freq *= 2
            print_freq *= 2
          if global_step == 16000:
            eval_freq *= 2
            print_freq *= 2

          scheduler.step()
          val_result = eval_running_model(val_dataloader)
          print('Global Step %d VAL res:\n' % global_step, val_result)
          log_wf.write('Global Step %d VAL res:\n' % global_step)
          log_wf.write(str(val_result) + '\n')

          if val_result['eval_loss'] < best_eval_loss:
            best_eval_loss = val_result['eval_loss']
            val_result['best_eval_loss'] = best_eval_loss
            # save model
            print('[Saving at]', state_save_path)
            log_wf.write('[Saving at] %s\n' % state_save_path)
            torch.save(model.state_dict(), state_save_path)
            output_config_file = os.path.join(output_dir, CONFIG_NAME)
            model.config.to_json_file(output_config_file)
            tokenizer.save_pretrained(output_dir)
        log_wf.flush()
        pass

    # add a eval step after each epoch
    scheduler.step()
    val_result = eval_running_model(val_dataloader)
    print('Epoch %d, Global Step %d VAL res:\n' % (epoch, global_step), val_result)
    log_wf.write('Global Step %d VAL res:\n' % global_step)
    log_wf.write(str(val_result) + '\n')

    if val_result['eval_loss'] < best_eval_loss:
      best_eval_loss = val_result['eval_loss']
      val_result['best_eval_loss'] = best_eval_loss
      # save model
      print('[Saving at]', state_save_path)
      log_wf.write('[Saving at] %s\n' % state_save_path)
      torch.save(model.state_dict(), state_save_path)
      output_config_file = os.path.join(output_dir, CONFIG_NAME)
      model.config.to_json_file(output_config_file)
      tokenizer.save_pretrained(output_dir)
    print(global_step, tr_loss / nb_tr_steps)
    log_wf.write('%d\t%f\n' % (global_step, tr_loss / nb_tr_steps))

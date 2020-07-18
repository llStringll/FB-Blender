import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel, DistilBertModel

def dot_attention(q, k, v, v_mask=None, dropout=None):
  attention_weights = torch.matmul(q, k.transpose(-1, -2))
  if v_mask is not None:
    attention_weights *= v_mask.unsqueeze(1)
  attention_weights = F.softmax(attention_weights, -1)
  if dropout is not None:
    attention_weights = dropout(attention_weights)
  output = torch.matmul(attention_weights, v)
  return output

class BertPolyModel(BertPreTrainedModel):
  def __init__(self, config, *inputs, **kwargs):
    super().__init__(config, *inputs, **kwargs)
    self.bert = kwargs['bert']

    self.vec_dim = 64

    self.poly_m = kwargs['poly_m']
    self.poly_code_embeddings = nn.Embedding(self.poly_m + 1, config.hidden_size)
    try:
      self.dropout = nn.Dropout(config.hidden_dropout_prob)
      self.context_fc = nn.Linear(config.hidden_size, self.vec_dim)
      self.response_fc = nn.Linear(config.hidden_size, self.vec_dim)
    except:
      self.dropout = nn.Dropout(config.dropout)
      self.context_fc = nn.Linear(config.dim, self.vec_dim)
      self.response_fc = nn.Linear(config.dim, self.vec_dim)

  def forward(self, context_input_ids, context_segment_ids, context_input_masks,
              responses_input_ids, responses_segment_ids, responses_input_masks, labels=None):
    ## only select the first response (whose lbl==1)
    if labels is not None:
      responses_input_ids = responses_input_ids[:, 0, :].unsqueeze(1)
      responses_segment_ids = responses_segment_ids[:, 0, :].unsqueeze(1)
      responses_input_masks = responses_input_masks[:, 0, :].unsqueeze(1)
    batch_size, res_cnt, seq_length = responses_input_ids.shape

    ## poly context encoder
    if isinstance(self.bert, DistilBertModel):
      state_vecs = self.bert(context_input_ids, context_input_masks)[-1]  # [bs, length, dim]
    else:
      state_vecs = self.bert(context_input_ids, context_input_masks, context_segment_ids)[0]  # [bs, length, dim]
    poly_code_ids = torch.arange(self.poly_m, dtype=torch.long, device=context_input_ids.device)
    poly_code_ids += 1
    poly_code_ids = poly_code_ids.unsqueeze(0).expand(batch_size, self.poly_m)
    poly_codes = self.poly_code_embeddings(poly_code_ids)
    context_vecs = dot_attention(poly_codes, state_vecs, state_vecs, context_input_masks, self.dropout)

    ## response encoder
    responses_input_ids = responses_input_ids.view(-1, seq_length)
    responses_input_masks = responses_input_masks.view(-1, seq_length)
    responses_segment_ids = responses_segment_ids.view(-1, seq_length)
    if isinstance(self.bert, DistilBertModel):
      state_vecs = self.bert(responses_input_ids, responses_input_masks)[-1]  # [bs, length, dim]
    else:
      state_vecs = self.bert(responses_input_ids, responses_input_masks, responses_segment_ids)[0]  # [bs, length, dim]
    poly_code_ids = torch.zeros(batch_size * res_cnt, 1, dtype=torch.long, device=context_input_ids.device)
    poly_codes = self.poly_code_embeddings(poly_code_ids)
    responses_vec = dot_attention(poly_codes, state_vecs, state_vecs, responses_input_masks, self.dropout)
    responses_vec = responses_vec.view(batch_size, res_cnt, -1)

    context_vecs = self.context_fc(self.dropout(context_vecs))
    context_vecs = F.normalize(context_vecs, 2, -1)  # [bs, m, dim]
    responses_vec = self.response_fc(self.dropout(responses_vec))
    responses_vec = F.normalize(responses_vec, 2, -1)

    ## poly final context vector aggregation
    if labels is not None:
      responses_vec = responses_vec.view(1, batch_size, -1).expand(batch_size, batch_size, self.vec_dim)
    final_context_vec = dot_attention(responses_vec, context_vecs, context_vecs, None, self.dropout)
    final_context_vec = F.normalize(final_context_vec, 2, -1)  # [bs, res_cnt, dim], res_cnt==bs when training

    dot_product = torch.sum(final_context_vec * responses_vec, -1)  # [bs, res_cnt], res_cnt==bs when training
    if labels is not None:
      mask = torch.eye(context_input_ids.size(0)).to(context_input_ids.device)
      loss = F.log_softmax(dot_product * 5, dim=-1) * mask
      loss = (-loss.sum(dim=1)).mean()
      cos_similarity = (dot_product + 1) / 2
      a=[]
      for _ in cos_similarity[:]:
          a.append(responses_input_ids[_.argmax()])
      a = torch.stack(a)
      return loss, a
    else:
      cos_similarity = (dot_product + 1) / 2
      return cos_similarity

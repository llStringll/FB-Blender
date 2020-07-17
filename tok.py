### train or fetch pretrained tokenizer, made to train on reddit pushshift.io
import os
import tokenizers
from tokenizers import Tokenizer
from tokenizers import ByteLevelBPETokenizer as bpe

class Tok:
	def __init__(self, train_corpus, save_file ,vocab_size=20000):
		if not os.path.exists(save_file+"/tokenizer.json"):
			if not os.path.exists(save_file):
				os.makedirs(save_file)
			print ("pretrained tokenizer not found, training now")
			self.tok = bpe(add_prefix_space=True, lowercase=True)
			self.tok.train([train_corpus], vocab_size=vocab_size, special_tokens=["[PAD]", "[CLS]", "[SEP]", "[UNK]"])
			self.tok.save(save_file+"/tokenizer.json")
		else:
			print ("pretrained tokenizer found, loading from file")
			self.tok=Tokenizer.from_file(save_file+"/tokenizer.json")

	def encode_plus(self, inpstr, max_length, truncate_at_max_length=True, pad_to_max_length=True, segment_ids=0):
		if truncate_at_max_length:
			self.tok.enable_truncation(max_length=max_length)
		inp_enc = self.tok.encode("[CLS]"+inpstr+"[SEP]")

		if pad_to_max_length and len(inp_enc.ids) < max_length:
			inp_ids = inp_enc.ids + [self.tok.token_to_id("[PAD]")]*(max_length - len(inp_enc.ids))
			inp_type_ids = [segment_ids]*(len(inp_ids))
			inp_attention_mask = inp_enc.attention_mask + [0]*(max_length - len(inp_enc.ids))

		return inp_ids, inp_type_ids, inp_attention_mask

	def decode(self, inp_ids):
		return self.tok.decode(inp_ids)


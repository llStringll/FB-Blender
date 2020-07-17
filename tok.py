### train or fetch pretrained tokenizer, made to train on reddit pushshift.io
import os
import tokenizers
from tokenizers import Tokenizer
from tokenizers import ByteLevelBPETokenizer as bpe

def get_tok(train_corpus,save_file,vocab_size=20000):
        if not os.path.exists(save_file+"/tokenizer.json"):
                if not os.path.exists(save_file):
                        os.makedirs(save_file)
                print ("pretrained tokenizer not found, training now")
                tok = bpe(add_prefix_space=True, lowercase=True)
                tok.train([train_corpus], vocab_size=vocab_size)
                tok.save(save_file+"/tokenizer.json")
        else:
                print ("pretrained tokenizer found, loading from file")
                tok=Tokenizer.from_file(save_file+"/tokenizer.json")
        return tok

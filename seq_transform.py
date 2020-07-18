class SelectionSequentialTransform(object):
  def __init__(self, tokenizer, max_len=128, max_history=10, pair_last=False):
    self.tokenizerPE = tokenizer[0]
    self.tokenizerGE = tokenizer[1]
    self.max_len = max_len
    self.max_history = max_history
    self.pair_last = pair_last

  def __call__(self, texts):
    input_ids_listPE, segment_ids_listPE, input_masks_listPE, contexts_masks_list = [], [], [], []
    input_ids_listGE, segment_ids_listGE, input_masks_listGE = [], [], []
    if self.max_history is not None:
      texts = texts[-self.max_history:]
    last_context = None
    if self.pair_last:
      last_context = texts[-1]
    for text in texts:
      tokenized_dictPE = self.tokenizerPE.encode_plus(text,
                                                  text_pair=last_context,
                                                  add_special_tokens=True,
                                                  max_length=self.max_len,
                                                  pad_to_max_length=True)
      input_idsPE, segment_idsPE, input_masksPE = tokenized_dictPE['input_ids'], tokenized_dictPE['token_type_ids'], \
                                            tokenized_dictPE['attention_mask']
      input_idsGE, segment_idsGE, input_masksGE = self.tokenizerGE.encode_plus(text,
                                                                            max_length=self.max_len,
                                                                            pad_to_max_length=True,
                                                                            segment_ids=segment_idsPE[0])
      assert len(input_idsPE) == self.max_len
      assert len(segment_idsPE) == self.max_len
      assert len(input_masksPE) == self.max_len
      assert len(input_idsGE) == self.max_len
      assert len(segment_idsGE) == self.max_len
      assert len(input_masksGE) == self.max_len
      input_ids_listPE.append(input_idsPE)
      segment_ids_listPE.append(segment_idsPE)
      input_masks_listPE.append(input_masksPE)
      input_ids_listGE.append(input_idsGE)
      segment_ids_listGE.append(segment_idsGE)
      input_masks_listGE.append(input_masksGE)
    contexts_masks_list = [1] * len(input_ids_listPE)
    if self.max_history is not None:
      tokenized_dict = self.tokenizer.encode_plus('',
                                                  text_pair='',
                                                  add_special_tokens=True,
                                                  max_length=self.max_len,
                                                  pad_to_max_length=True)
      input_ids, segment_ids, input_masks = tokenized_dict['input_ids'], tokenized_dict['token_type_ids'], \
                                            tokenized_dict['attention_mask']
      for _ in range(self.max_history - len(texts)):
        input_ids_list.append(input_ids[:])
        segment_ids_list.append(segment_ids[:])
        input_masks_list.append(input_masks[:])
      contexts_masks_list += [0] * (self.max_history - len(texts))

    return input_ids_listPE, segment_ids_listPE, input_masks_listPE, input_ids_listGE, segment_ids_listGE, input_masks_listGE, contexts_masks_list

  def __str__(self) -> str:
    return 'maxlen%d_maxhistory%d_pairlast%s' % (self.max_len, self.max_history, str(self.pair_last))


class SelectionJoinTransform(object):
  def __init__(self, tokenizer, max_len=512, max_history=10):
    self.tokenizerPE = tokenizer[0]
    self.tokenizerGE = tokenizer[1]
    self.max_len = max_len
    self.max_history = max_history

    self.cls_id = self.tokenizerPE.convert_tokens_to_ids(['[CLS]'])[0]
    self.sep_id = self.tokenizerPE.convert_tokens_to_ids(['[SEP]'])[0]
    self.pad_id = 0

  def __call__(self, texts):
    input_ids_listPE, segment_ids_listPE, input_masks_listPE = [], [], []
    input_ids_listGE, segment_ids_listGE, input_masks_listGE = [], [], []

    for text,b in texts[::-1][:self.max_history]:
      tokenized_dictPE = self.tokenizerPE.encode_plus(text,
                                                  text_pair=None,
                                                  add_special_tokens=True,
                                                  max_length=self.max_len,
                                                  pad_to_max_length=False)
      input_idsPE, input_masksPE = tokenized_dictPE['input_ids'], tokenized_dictPE['attention_mask']

      if "your persona:" in text:
          segment_idsPE = [1] * len(input_idsPE) # segment id for presona
      elif b==False:
          segment_idsPE = [1] * len(input_idsPE) # segment id for speaker speech
      else:
          segment_idsPE = [0] * len(input_idsPE) # segment id for past bot speech, consistent with segment id assigned to responses

      input_idsGE, segment_idsGE, input_masksGE = self.tokenizerGE.encode_plus(text,
                                                                            max_length = self.nax_len,
                                                                            segment_ids = segment_idsPE[0],
                                                                            pad_to_max_length = False)

      if len(input_ids_listGE) > 0:
        input_idsGE = input_idsGE[1:]
        segment_idsGE = segment_idsGE[1:]
        input_masksGE = input_masksGE[1:]
      input_ids_listGE.extend(input_idsGE)
      segment_ids_listGE.extend(segment_idsGE)
      input_masks_listGE.extend(input_masksGE)

      if len(input_ids_listPE) > 0:
        input_idsPE = input_idsPE[1:]
        segment_idsPE = segment_idsPE[1:]
        input_masksPE = input_masksPE[1:]
      input_ids_listPE.extend(input_idsPE)
      segment_ids_listPE.extend(segment_idsPE)
      input_masks_listPE.extend(input_masksPE)

      if len(input_ids_listPE) >= self.max_len and len(input_ids_listGE) >= self.max_len:
        input_ids_listPE = input_ids_listPE[:self.max_len - 1] + [self.sep_id]
        segment_ids_listPE = segment_ids_listPE[:self.max_len]
        input_masks_listPE = input_masks_listPE[:self.max_len]

        input_ids_listGE = input_ids_listGE[:self.max_len - 1] + [self.sep_id]
        segment_ids_listGE = segment_ids_listGE[:self.max_len]
        input_masks_listGE = input_masks_listGE[:self.max_len]
        break

      if len(input_ids_listPE) >= self.max_len:
        input_ids_listPE = input_ids_listPE[:self.max_len - 1] + [self.sep_id]
        segment_ids_listPE = segment_ids_listPE[:self.max_len]
        input_masks_listPE = input_masks_listPE[:self.max_len]
        break

      if len(input_ids_listGE) >= self.max_len:
        input_ids_listGE = input_ids_listGE[:self.max_len - 1] + [self.sep_id]
        segment_ids_listGE = segment_ids_listGE[:self.max_len]
        input_masks_listGE = input_masks_listGE[:self.max_len]
        break

    input_ids_listPE += [self.pad_id] * (self.max_len - len(input_ids_listPE))
    segment_ids_listPE += [0] * (self.max_len - len(segment_ids_listPE))
    input_masks_listPE += [0] * (self.max_len - len(input_masks_listPE))

    input_ids_listGE += [self.tokenizerGE.token_to_id("[PAD]")] * (self.max_len - len(input_ids_listGE))
    segment_ids_listGE += [0] * (self.max_len - len(segment_ids_listGE))
    input_masks_listGE += [0] * (self.max_len - len(input_masks_listGE))

    assert len(input_ids_listPE) == self.max_len
    assert len(segment_ids_listPE) == self.max_len
    assert len(input_masks_listPE) == self.max_len
    assert len(input_ids_listGE) == self.max_len
    assert len(segment_ids_listGE) == self.max_len
    assert len(input_masks_listGE) == self.max_len

    return input_ids_listPE, segment_ids_listPE, input_masks_listPE, input_ids_listGE, segment_ids_listGE, input_masks_listGE

  def __str__(self) -> str:
    return '[join_str]maxlen%d_maxhis%d' % (self.max_len, self.max_history)

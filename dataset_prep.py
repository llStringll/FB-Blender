import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import os
from common_utils import pickle_load, pickle_dump

class SelectionDataset(Dataset):
    def __init__(self, file_path, context_transform, response_transform, sample_cnt=None):
        self.context_transform = context_transform
        self.response_transform = response_transform
        self.data_source = []
        with open(file_path, encoding='utf-8') as f:
            r_lines = f.read().splitlines()
        context = []
        for i, line in enumerate(r_lines):
            index, line = line.split(' ', maxsplit = 1)
            if line.find('your persona:') == 0:
                context.append((line, False))
            else:
                req_res, responses = line.split('\t\t')
                request, response = req_res.split('\t')
                context.append((request, False))
                if len(context) != 1:
                    start = len(self.data_source)
                self.data_source.append({
                    'start': start,
                    'index': int(index),
                    'context': context,
                    'response': [(response, True)],
                    'responses': responses.split('|'),
                    'labels': [int(response == r) for r in responses.split('|')]
                })
                context = []
            if sample_cnt is not None and len(self.data_source) >= sample_cnt:
              break                
    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self.__get_single_item__(index) for index in indices]
        return self.__get_single_item__(indices)

    def __get_single_item__(self, index):
        group = self.data_source[index]
        context = []
        for i in range(group['start'], index):
            context += self.data_source[i]['context']
            context += self.data_source[i]['response']
        context += self.data_source[index]['context']
        responses, labels = group['responses'], group['labels']
        transformed_responses = self.response_transform(responses[::-1])
        transformed_context = self.context_transform(context)
        labels = labels[::-1]
        key_data = transformed_context, transformed_responses, labels
        return key_data

    def batchify(self, batch):
      contexts_token_ids_list_batch, contexts_segment_ids_list_batch, contexts_input_masks_list_batch, contexts_masks_batch, \
      responses_token_ids_list_batch, responses_segment_ids_list_batch, responses_input_masks_list_batch = [], [], [], [], [], [], []
      labels_batch = []
      for sample in batch:
        (contexts_token_ids_list, contexts_segment_ids_list, contexts_input_masks_list, contexts_masks_list), \
        (responses_token_ids_list, responses_segment_ids_list, responses_input_masks_list, _) = sample[:2]

        contexts_token_ids_list_batch.append(contexts_token_ids_list)
        contexts_segment_ids_list_batch.append(contexts_segment_ids_list)
        contexts_input_masks_list_batch.append(contexts_input_masks_list)
        contexts_masks_batch.append(contexts_masks_list)

        responses_token_ids_list_batch.append(responses_token_ids_list)
        responses_segment_ids_list_batch.append(responses_segment_ids_list)
        responses_input_masks_list_batch.append(responses_input_masks_list)

        labels_batch.append(sample[-1])

      long_tensors = [contexts_token_ids_list_batch, contexts_segment_ids_list_batch, contexts_input_masks_list_batch,
                      contexts_masks_batch,
                      responses_token_ids_list_batch, responses_segment_ids_list_batch, responses_input_masks_list_batch]

      contexts_token_ids_list_batch, contexts_segment_ids_list_batch, contexts_input_masks_list_batch, contexts_masks_batch, \
      responses_token_ids_list_batch, responses_segment_ids_list_batch, responses_input_masks_list_batch = (
        torch.tensor(t, dtype=torch.long) for t in long_tensors)

      labels_batch = torch.tensor(labels_batch, dtype=torch.long)
      return contexts_token_ids_list_batch, contexts_segment_ids_list_batch, contexts_input_masks_list_batch, contexts_masks_batch, \
             responses_token_ids_list_batch, responses_segment_ids_list_batch, responses_input_masks_list_batch, labels_batch

    def batchify_join_str(self, batch):
      contexts_token_ids_list_batch, contexts_segment_ids_list_batch, contexts_input_masks_list_batch, \
      responses_token_ids_list_batch, responses_segment_ids_list_batch, responses_input_masks_list_batch = [], [], [], [], [], []
      labels_batch = []
      for sample in batch:
        (contexts_token_ids_list, contexts_segment_ids_list, contexts_input_masks_list), \
        (responses_token_ids_list, responses_segment_ids_list, responses_input_masks_list, _) = sample[:2]

        contexts_token_ids_list_batch.append(contexts_token_ids_list)
        contexts_segment_ids_list_batch.append(contexts_segment_ids_list)
        contexts_input_masks_list_batch.append(contexts_input_masks_list)

        responses_token_ids_list_batch.append(responses_token_ids_list)
        responses_segment_ids_list_batch.append(responses_segment_ids_list)
        responses_input_masks_list_batch.append(responses_input_masks_list)

        labels_batch.append(sample[-1])

      long_tensors = [contexts_token_ids_list_batch, contexts_segment_ids_list_batch, contexts_input_masks_list_batch,
                      responses_token_ids_list_batch, responses_segment_ids_list_batch, responses_input_masks_list_batch]

      contexts_token_ids_list_batch, contexts_segment_ids_list_batch, contexts_input_masks_list_batch, \
      responses_token_ids_list_batch, responses_segment_ids_list_batch, responses_input_masks_list_batch = (
        torch.tensor(t, dtype=torch.long) for t in long_tensors)

      labels_batch = torch.tensor(labels_batch, dtype=torch.long)
      return contexts_token_ids_list_batch, contexts_segment_ids_list_batch, contexts_input_masks_list_batch, \
             responses_token_ids_list_batch, responses_segment_ids_list_batch, responses_input_masks_list_batch, labels_batch

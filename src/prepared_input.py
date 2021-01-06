import torch
import numpy as np
import os
import random
from transformers import RobertaTokenizer


class Prepared:

    def __init__(self, name, df, max_length=0):
        self.name = name
        self.df = df
        self.metaphor_labels = torch.tensor(self.df['metaphor'].values, dtype=torch.int64)
        self.length = None
        self.global_input_ids = None
        self.global_attention_mask = None
        self.global_token_type_ids = None
        self.local_input_ids = None
        self.local_attention_mask = None
        self.local_token_type_ids = None

        # See if we have cached a set of inputs for our model and use that, otherwise do the necessary work
        if self.check_prepared(name) and (
                max_length == 0 or
                len(torch.load('../data/intermediate/' + name + '_global_input_ids.pt')[0]) == max_length):
            self.load_prepared()
        else:
            self.prepare_inputs(max_length)
            self.save_prepared()

    # Save a prepared object so that we don't have repeat work
    def save_prepared(self):
        torch.save(self.global_input_ids, '../data/intermediate/' + self.name + '_global_input_ids.pt')
        torch.save(self.global_attention_mask, '../data/intermediate/' + self.name + '_global_attention_mask.pt')
        torch.save(self.global_token_type_ids, '../data/intermediate/' + self.name + '_global_token_type_ids.pt')
        torch.save(self.local_input_ids, '../data/intermediate/' + self.name + '_local_input_ids.pt')
        torch.save(self.local_attention_mask, '../data/intermediate/' + self.name + '_local_attention_mask.pt')
        torch.save(self.local_token_type_ids, '../data/intermediate/' + self.name + '_local_token_type_ids.pt')

    # Load a prepared object so that we don't have repeat work
    def load_prepared(self):
        self.global_input_ids = torch.load('../data/intermediate/' + self.name + '_global_input_ids.pt')
        self.global_attention_mask = torch.load('../data/intermediate/' + self.name + '_global_attention_mask.pt')
        self.global_token_type_ids = torch.load('../data/intermediate/' + self.name + '_global_token_type_ids.pt')
        self.local_input_ids = torch.load('../data/intermediate/' + self.name + '_local_input_ids.pt')
        self.local_attention_mask = torch.load('../data/intermediate/' + self.name + '_local_attention_mask.pt')
        self.local_token_type_ids = torch.load('../data/intermediate/' + self.name + '_local_token_type_ids.pt')
        self.length = len(self.global_input_ids[0])

    @staticmethod
    def check_prepared(name):
        filename_ends = ['_global_input_ids.pt', '_global_attention_mask.pt', '_global_token_type_ids.pt',
                         '_local_input_ids.pt', '_local_attention_mask.pt', '_local_token_type_ids.pt']
        return all([os.path.exists('../data/intermediate/' + str(name) + end) for end in filename_ends])

    def prepare_inputs(self, max_length):
        global_context_input = self.df['sentence'] + " </s> " + self.df['query'] + " </s> " + self.df['pos'] + " </s> "\
            + self.df['fgpos']
        local_context_input = self.df['local'] + " </s> " + self.df['query'] + " </s> " + self.df['pos'] + " </s> " \
            + self.df['fgpos']

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        # We've added the intermediate separator tokens, but let the tokenizer handle the start, end and padding tokens
        if max_length <= 0:
            global_context_tokenized = tokenizer(list(global_context_input),
                                                 add_special_tokens=True, padding=True, return_tensors='pt')
            local_context_tokenized = tokenizer(list(local_context_input),
                                                add_special_tokens=True, padding='max_length', return_tensors='pt',
                                                max_length=global_context_tokenized['input_ids'].size()[1])
        else:
            global_context_tokenized = tokenizer(list(global_context_input),
                                                 add_special_tokens=True, padding='max_length', return_tensors='pt',
                                                 max_length=max_length)
            local_context_tokenized = tokenizer(list(local_context_input),
                                                add_special_tokens=True, padding='max_length', return_tensors='pt',
                                                max_length=max_length)

        self.global_input_ids = global_context_tokenized['input_ids']
        self.global_attention_mask = global_context_tokenized['attention_mask']
        self.global_token_type_ids = torch.zeros(global_context_tokenized['input_ids'].size(), dtype=torch.long)
        self.local_input_ids = local_context_tokenized['input_ids']
        self.local_attention_mask = local_context_tokenized['attention_mask']
        self.local_token_type_ids = torch.zeros(local_context_tokenized['input_ids'].size(), dtype=torch.long)
        self.length = len(self.global_input_ids[0])

    # 0 = metaphor, 1 = input_ids_a, 2 = att_mask_a, 3 = tok_type_ids_a, 4 = input_ids_b, 5 = att_mask_b,
    # 6 = tok_type_ids_b
    def get_tensors(self, index=None):
        if index is None:
            return self.metaphor_labels, self.global_input_ids, self.global_attention_mask, self.global_token_type_ids,\
               self.local_input_ids, self.local_attention_mask, self.local_token_type_ids
        else:
            return self.metaphor_labels[index], self.global_input_ids[index], self.global_attention_mask[index], \
                   self.global_token_type_ids[index], self.local_input_ids[index], self.local_attention_mask[index], \
                   self.local_token_type_ids[index]

    def to_device(self, device):
        self.metaphor_labels = self.metaphor_labels.to(device)
        self.global_input_ids = self.global_input_ids.to(device)
        self.global_attention_mask = self.global_attention_mask.to(device)
        self.global_token_type_ids = self.global_token_type_ids.to(device)
        self.local_input_ids = self.local_input_ids.to(device)
        self.local_attention_mask = self.local_attention_mask.to(device)
        self.local_token_type_ids = self.local_token_type_ids.to(device)

    def to_folds(self, num_folds):
        # Split into metaphor and literal for stratified cross validation
        metaphor_indices = list(np.where(self.metaphor_labels.cpu().numpy() == 1)[0])
        literal_indices = list(np.where(self.metaphor_labels.cpu().numpy() == 0)[0])

        met_size = len(metaphor_indices) // num_folds
        lit_size = len(literal_indices) // num_folds

        met_rem = len(metaphor_indices) % num_folds
        lit_rem = len(literal_indices) % num_folds

        folds_indices = [[] for _ in range(num_folds)]
        for i in range(num_folds):
            # By starting from the back with the literals, we can make sure that the longest metaphor and literal folds
            # don't get added together
            folds_indices[i].extend(random.sample(metaphor_indices, k=met_size + (1 if met_rem > 0 else 0)))
            folds_indices[num_folds-i-1].extend(random.sample(literal_indices, k=lit_size + (1 if lit_rem > 0 else 0)))

            met_rem -= (1 if met_rem > 0 else 0)
            lit_rem -= (1 if lit_rem > 0 else 0)

        [random.shuffle(fold) for fold in folds_indices]
        return [self.get_tensors(fold) for fold in folds_indices]

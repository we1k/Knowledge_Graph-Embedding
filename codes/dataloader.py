import logging
import os
import time

import numpy as np
import torch
from scipy import sparse
from torch.utils.data import Dataset

def time_it(fn):
    def wrapper(*args, **kwargs):
        start = time.time()
        ret = fn(*args, **kwargs)
        end = time.time()
        print(f'Time: {end - start}')
        return ret

    return wrapper

class TrainDataset(Dataset):

    def _get_adj_mat(self):
        a_mat = sparse.dok_matrix((self.))
    
    @time_it
    def build_k_hop(self, k_hop, dataset_name):
        pass



    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode, k_hop, n_rw, dsn):
        self.len = len(triples)
        self.triples = triples
        self.triples_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)
        self.dsn = dsn.split('/')[1]  # dataset name

    def __len__(self):
        return self.len

    
    def __getitem__(self, idx):
        pass

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode

    
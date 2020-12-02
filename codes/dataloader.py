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
        a_mat = sparse.dok_matrix((self.nentity, self.nentity))
        for (h, _, t) in self.triples:
            a_mat[t, h] = 1
            a_mat[h, t] = 1
    
        a_mat = a_mat.tocsr()
        return a_mat

    @time_it
    def build_k_hop(self, k_hop, dataset_name):
        if k_hop == 0:
            return None
        
        save_path = f'cached_matrices/matrix_{dataset_name}_k{k_hop}_nrw0.npz'

        if os.path.exists(save_path):
            logging.info(f'Using cache from {save_path}')
            k_mat = sparse.load_npz(save_path)
            return k_mat

        a_mat = self._get_adj_mat()
        _k_mat = a_mat ** (k_hop - 1)
        k_mat = _k_mat * a_mat + _k_mat

        sparse.save_npz(save_path, k_mat)

        return k_mat
        
    @time_it
    def build_k_rw(self, n_rw, k_hop, dataset_name):
        if n_rw == 0 or k_hop == 0:
            return None

        save_path = f'cached_matrices/matrix_{dataset_name}_k{k_hop}_nrw{n_rw}.npz'

        if os.path.exists(save_path):
            logging.info(f'Using cache from {save_path}')
            k_mat = sparse.load_npz(save_path)
            return k_mat

        a_mat = self._get_adj_mat()
        k_mat = sparse.dok_matrix((self.nentity, self.nentity))

        randomly_sampled = 0

        for i in range(0, self.nentity):
            neighbors = a_mat[i]
            if len(neighbors.indices) == 0:
                randomly_sampled += 1
                walker = np.random.randint(self.nentity, size=n_rw)
                k_mat[i, walker] = 1
            else:
                for _ in range(n_rw):
                    walker = i
                    for _ in range(0, k_hop):
                        idx = np.random.randint(len(neighbors.indices))
                        walker = neighbors.indices[idx]
                        neighbors = a_mat[walker]
                    k_mat[i, walker] += 1
        logging.info(f'randomly_sampled: {randomly_sampled}')
        k_mat = k_mat.tocsr()
        
        sparse.save_npz(save_path, k_mat)

        return k_mat

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

def main():

    dataset = TrainDatas()


if __name__ == "__main__":
    main()
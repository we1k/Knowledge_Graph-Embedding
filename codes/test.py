import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn

from scipy import sparse

from dataloader import TrainDataset

def set_logger():
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    # console = logging.StreamHandler()
    # console.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    # console.setFormatter(formatter)

def read_triple(data_path, entity2id, relation2id):
    triples = []
    with open(data_path) as f:
        lines = f.readlines()
        for line in lines:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def time_it(fn):
    def wrapper(*args, **kwargs):
        start = time.time()
        ret = fn(*args, **kwargs)
        end = time.time()
        logging.info(f'time:{end - start}')
        return ret

    return wrapper

def main():
    # logging setting
    set_logger()

    # config
    data_path = './data/wn18rr/'
    entity2id = dict()
    relation2id = dict()
    with open(os.path.join(data_path, 'entities.dict')) as f:
        lines = f.readlines()
        for line in lines:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)
    with open(os.path.join(data_path, 'relations.dict')) as f:
        lines = f.readlines()
        for line in lines:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
    nrelation = len(relation2id)
    nentity = len(entity2id)

    train_triples = read_triple(os.path.join(data_path, 'train.txt'), entity2id, relation2id)
    test_triples = read_triple(os.path.join(data_path, 'test.txt'), entity2id, relation2id)
    
    def _get_adj_mat():
        a = sparse.dok_matrix((3, 3))
        a[0, 1] = 1
        a[1, 0] = 1
        a[1, 2] = 1
        a[2, 1] = 1
        a = a.tocsr()
        return a
    
    @time_it
    def build_k_hop(k_hop, dataset_name):
        if k_hop == 0:
            return None

        save_path = f'cached_matrices/matrix_{dataset_name}_k{k_hop}_nrw0.npz'

        if os.path.exists(save_path):
            logging.info(f'Using cached matrix: {save_path}')
            k_mat = sparse.load_npz(save_path)
            print(k_mat.todense())
            return k_mat
       
        # K = S^+(Ak + Ak-1)
        _a_mat = _get_adj_mat()
        _k_mat = _a_mat ** (k_hop - 1)
        k_mat = _k_mat * _a_mat + _k_mat

        sparse.save_npz(save_path, k_mat)
        
        logging.info(k_mat.todense())
        return k_mat

    @time_it
    def build_k_rw(n_rw, k_hop, dataset_name):
        '''
        n_rw: number of negative random walk
        return matrix
        '''
        if n_rw == 0 or k_hop == 0:
            return None
        save_path = f'cached_matrices/matrix_{dataset_name}_k{k_hop}_nrw{n_rw}.npz'

        if os.path.exists(save_path):
            logging.info(f'Using cached matrix: {save_path}')
            k_mat = sparse.load_npz(save_path)
            return k_mat

        a_mat = _get_adj_mat()
        k_mat = sparse.dok_matrix((3, 3))

        randomly_sampled = 0

        for i in range(0, 3):
            neighbors = a_mat[i]
            if len(neighbors.indices) == 0:
            # which is a outlier, random switch to another node
                randomly_sampled += 1
                walker = np.random.randint(3, size=n_rw)
                k_mat[i, walker] = 1
            else:
                for _ in range(0, n_rw):
                    for _ in range(0, k_hop):
                        idx = np.random.randint(len(neighbors.indices))
                        walker = neighbors.indices[idx]
                        neighbors = a_mat[walker]
                    k_mat[i, walker] += 1
        logging.info(f'randomly_sampled: {randomly_sampled}')
        k_mat = k_mat.tocsr()
        
        sparse.save_npz(save_path, k_mat)
        print(k_mat.todense())

        return k_mat



if __name__ == "__main__":
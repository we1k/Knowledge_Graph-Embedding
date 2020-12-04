import logging
import os
import time
import numpy as np
from scipy import sparse

def _get_adj_mat():
    a_mat = sparse.dok_matrix((nentity, nentity))
    for (h, _, t) in triples:
        a_mat[t, h] = 1
        a_mat[h, t] = 1

    a_mat = a_mat.tocsr()
    return a_mat

def build_k_hop(k_hop, dataset_name):
    if k_hop == 0:
        return None

    save_path = f'cached_matrices/matrix_{dataset_name}_k{k_hop}_nrw0.npz'

    if os.path.exists(save_path):
        k_mat = sparse.load_npz(save_path)
        return k_mat

    _a_mat = _get_adj_mat()
    _k_mat = _a_mat ** (k_hop - 1)
    k_mat = _k_mat * _a_mat + _k_mat

    sparse.save_npz(save_path, k_mat)

    return k_mat

def build_k_rw(n_rw, k_hop, dataset_name):

    if n_rw == 0 or k_hop == 0:
        return None

    save_path = f'cached_matrices/matrix_{dataset_name}_k{k_hop}_nrw{n_rw}.npz'

    if os.path.exists(save_path):
        k_mat = sparse.load_npz(save_path)
        return k_mat

    a_mat = _get_adj_mat()
    k_mat = sparse.dok_matrix((nentity, nentity))

    randomly_sampled = 0

    for i in range(0, nentity):
        neighbors = a_mat[i]
        if len(neighbors.indices) == 0:
            randomly_sampled += 1
            walker = np.random.randint(nentity, size=n_rw)
            k_mat[i, walker] = 1
        else:
            for _ in range(0, n_rw):
                walker = i
                for _ in range(0, k_hop):
                    idx = np.random.randint(len(neighbors.indices))
                    walker = neighbors.indices[idx]
                    neighbors = a_mat[walker]
                k_mat[i, walker] += 1
    k_mat = k_mat.tocsr()

    sparse.save_npz(save_path, k_mat)

    return k_mat

def main():
    dataset_name = 'wn18rr'
    data_path = 'data/'+dataset_name+'/'
    with open(os.path.join(data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)


    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)

    nentity = len(entity2id)
    nrelation = len(relation2id)

    n_rw = int(input('n_rw:'))
    k_hop = int(input('k_hop'))

    if n_rw == 0:
        build_k_hop(k_hop, dataset_name)
    else:
        build_k_rw(k_hop,n_rw,dataset_name)


if __name__ == "__main__":
    main()
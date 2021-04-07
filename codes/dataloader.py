import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from torch.utils.data import Dataset, WeightedRandomSampler

def time_it(fn):
    def wrapper(*args, **kwargs):
        start = time.time()
        ret = fn(*args, **kwargs)
        end = time.time()
        logging.info(f'Time: {end - start}')
        return ret

    return wrapper


class TrainDataset(Dataset):

    def _get_adj_mat(self, dataset_name):

        a_save_path = f'cached_matrices/matrix_{dataset_name}_a.npz'
        r_save_path = f'cached_matrices/matrix_{dataset_name}_r.npz'

        if os.path.exists(a_save_path) and os.path.exists(a_save_path):
            a_mat = sparse.load_npz(a_save_path)
            r_mat = sparse.load_npz(r_save_path)
            return a_mat, r_mat

        a_mat = sparse.dok_matrix((self.nentity, self.nentity))
        r_mat = sparse.dok_matrix((self.nentity, self.nrelation))
        for (h, r, t) in self.triples:
            a_mat[t, h] = 1
            a_mat[h, t] = 1
            r_mat[h, r] = 1
            r_mat[t, r]= 1

        a_mat = a_mat.tocsr()
        r_mat = r_mat.tocsr()
        sparse.save_npz(a_save_path, a_mat)
        sparse.save_npz(r_save_path, r_mat)
        return a_mat, r_mat

    @time_it
    def build_weighted_rw(self, n_rw, k_hop, dataset_name):
        if n_rw == 0 or k_hop == 0:
            return None

        save_path = f'cached_matrices/{dataset_name}_rw{n_rw}_hop{k_hop}_step{self.step}.npy'
        
        if os.path.exists(save_path):
            logging.info(f'Using cached matrix')
            false_entity = np.load(save_path, allow_pickle=True).item()
            return false_entity

        # USING A MATRIX TO STORE THE RELATION
        a_mat, r_mat = self._get_adj_mat(dataset_name)

        randomly_sampled = 0

        model_func = {
            'TransE': self.model.TransE,
            'DistMult': self.model.DistMult,
            'ComplEx': self.model.ComplEx,
            'RotatE': self.model.RotatE,
            'pRotatE': self.model.pRotatE
        }

        false_entity = {}
        for i in range(0, self.nentity):
            neighbors = a_mat[i]
            relations = r_mat[i]
            if len(neighbors.indices) == 0:
                randomly_sampled += 1
                e_walker = np.random.randint(self.nentity, size=n_rw)
                r_walker = np.random.randint(self.nrelation, size=n_rw)
                for j in range(0, n_rw):
                    if (i, r_walker[i]) not in false_entity:
                        false_entity[(i, r_walker[i])] = []
                    false_entity[(i, r_walker[i])].append(e_walker[i])
            else:
                head = self.model.entity_embedding[i].view(1, 1, -1)
                for r_idx in relations.indices:
                    relation = self.model.relation_embedding[r_idx].view(1, 1, -1)
                    for _ in range(0, n_rw):
                        walker = i
                        for _ in range(0, k_hop):
                            score = []
                            for t_idx in neighbors.indices:
                                tail = self.model.entity_embedding[t_idx].view(1, 1, -1)
                                single_score = model_func[self.model.model_name](head, relation, tail, 'single')
                                score.append(single_score.item())
                            weight = F.softmax(torch.Tensor(score), dim=0)
                            idx = list(WeightedRandomSampler(weight, 1))[0]
                            walker = neighbors.indices[idx]
                            neighbors = a_mat[walker]
                        if (i, r_idx) not in false_entity:
                            false_entity[(i, r_idx)] = []
                        false_entity[(i, r_idx)].append(walker)

        for entity, relation in false_entity:
            false_entity[(entity, relation)] = np.array(list(set(false_entity[(entity, relation)])))

        logging.info(f'randomly_sampled: {randomly_sampled}')

        np.save(save_path, false_entity)

        return false_entity
    
    @time_it
    def build_unweighted_rw(self, n_rw, k_hop, dataset_name):
        """
        Returns:
            k_mat: sparse |V| * |V| adjacency matrix
        """
        if n_rw == 0 or k_hop == 0:
            return None

        save_path = f'cached_matrices/matrix_{dataset_name}_k{k_hop}_nrw{n_rw}.npz'

        if os.path.exists(save_path):
            logging.info(f'Using cached matrix: {save_path}')
            k_mat = sparse.load_npz(save_path)
            return k_mat

        a_mat, r_mat = self._get_adj_mat(dataset_name)
        k_mat = sparse.dok_matrix((self.nentity, self.nentity))

        randomly_sampled = 0

        for i in range(0, self.nentity):
            neighbors = a_mat[i]
            if len(neighbors.indices) == 0:
                randomly_sampled += 1
                walker = np.random.randint(self.nentity, size=n_rw)
                k_mat[i, walker] = 1
            else:
                for _ in range(0, n_rw):
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

    def __init__(self, triples, model, step, negative_sample_size, mode, k_hop, n_rw, dsn):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.model = model
        self.step = step
        self.nentity = self.model.nentity
        self.nrelation = self.model.nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)
        self.dsn = dsn.split('/')[1]  # dataset name
        if self.step == 0:
            self.k_neighbors = self.build_unweighted_rw(n_rw=n_rw, k_hop=k_hop, dataset_name=self.dsn)
        else:
            self.k_neighbors = self.build_weighted_rw(n_rw=30, k_hop=k_hop, dataset_name=self.dsn)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        head, relation, tail = positive_sample

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation - 1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        negative_sample_list = []
        negative_sample_size = 0

        k_hop_flag = True

        if self.step == 0:
            while negative_sample_size < self.negative_sample_size:
                if self.k_neighbors is not None and k_hop_flag:
                    if self.mode == 'head-batch':
                        khop = self.k_neighbors[tail].indices
                    elif self.mode == 'tail-batch':
                        khop = self.k_neighbors[head].indices
                    else:
                        raise ValueError('Training batch mode %s not supported' % self.mode)
                    negative_sample = khop[np.random.randint(len(khop), size=self.negative_sample_size * 2)].astype(
                        np.int64)
                else:
                    negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
                if self.mode == 'head-batch':
                    mask = np.in1d(
                        negative_sample,
                        self.true_head[(relation, tail)],
                        assume_unique=True,
                        invert=True
                    )
                elif self.mode == 'tail-batch':
                    mask = np.in1d(
                        negative_sample,
                        self.true_tail[(head, relation)],
                        assume_unique=True,
                        invert=True
                    )
                else:
                    raise ValueError('Training batch mode %s not supported' % self.mode)
                negative_sample = negative_sample[mask]
                negative_sample_list.append(negative_sample)
                if negative_sample.size == 0:
                    k_hop_flag = False
                negative_sample_size += negative_sample.size

        else:
            while negative_sample_size < self.negative_sample_size:
                if self.k_neighbors is not None and k_hop_flag:
                    if self.mode == 'head-batch':
                        khop = self.k_neighbors[(tail, relation)]
                    elif self.mode == 'tail-batch':
                        khop = self.k_neighbors[(head, relation)]
                    else:
                        raise ValueError('Training batch mode %s not supported' % self.mode)
                    negative_sample = khop[np.random.randint(len(khop), size=self.negative_sample_size * 2)].astype(
                        np.int64)
                else:
                    negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
                if self.mode == 'head-batch':
                    mask = np.in1d(
                        negative_sample,
                        self.true_head[(relation, tail)],
                        assume_unique=True,
                        invert=True
                    )
                elif self.mode == 'tail-batch':
                    mask = np.in1d(
                        negative_sample,
                        self.true_tail[(head, relation)],
                        assume_unique=True,
                        invert=True
                    )
                else:
                    raise ValueError('Training batch mode %s not supported' % self.mode)
                negative_sample = negative_sample[mask]
                negative_sample_list.append(negative_sample)
                if negative_sample.size == 0:
                    k_hop_flag = False
                negative_sample_size += negative_sample.size


        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

        negative_sample = torch.from_numpy(negative_sample)

        positive_sample = torch.LongTensor(positive_sample)

        return positive_sample, negative_sample, subsampling_weight, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode

    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation - 1) not in count:
                count[(tail, -relation - 1)] = start
            else:
                count[(tail, -relation - 1)] += 1
        return count

    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''

        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

        return true_head, true_tail


class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)

        tmp = torch.LongTensor(tmp)
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))

        return positive_sample, negative_sample, filter_bias, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode


class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data

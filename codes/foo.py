import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy import sparse
import time

class Mydata(Dataset):
    def __init__(self, weighted, model):
        super().__init__()
        self.x = torch.randn(1000, 3)
        self.size = self.x.size(0)
        self.weighted = weighted
        self.model = model
        
    def __getitem__(self, index):
        if self.weighted:
            self.model.foo()
            return self.x[index]
            
        else:
            return self.x[index]
    def __len__(self):
        return self.size

class iterator(object):
    def __init__(self, dataloader):
        self.iter = self.one_shot_iterator(dataloader)

    def __next__(self):
        data = next(self.iter)

    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data

class model():
    def __init__(self):
        self.name = 'fccc'
        self.val = torch.randn(4, 3)
        self.epsilon = nn.Parameter(
            torch.Tensor([3]),
            requires_grad=False
        )

    def foo(self):
        print('worked')
        print(self.val, self.name)

    def forward(self, sample):
        head = sample[:, 0]
        relation = sample[:, 1]
        tail = sample[:,2]
        score = self.cal(head, relation, tail)
        return score

    def cal(self, head, tail, relation):
        score = head + relation - tail
        return score

    def __str__(self):
        return "fcuk"


def main():
    # foo = model()
    # data = Mydata(True, foo)
    # data_loader = DataLoader(data, batch_size=5, shuffle=True)
    # iterator = iter(data_loader)
    
    # for i in range(10):
    #     score = foo.forward(next(iterator))
    #     print(score)
    pass

if __name__ == "__main__":
    # main()
    # data = TrainDataset(
    #     train_triples,
    #     nentity,
    #     nrelation,
    #     args.negative_sample_size,
    #     'head-batch',
    #     args.negative_k_hop_sampling,
    #     args.negative_n_random_walks,
    #     dsn=args.data_path)

    a = sparse.dok_matrix((50, 50))
    print(a)
    # a[1, 1] = 1
    # a1 = a.tocsr()
    # a[1, 0] = 2
    # a2 = a.tocsr()
    # sparse.save_npz('tet.npz', a1)
    # sparse.save_npz('tet.npz', a2)

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
                        weight = F.softmax(torch.Tensor(weight), dim=0)
                        idx = list(WeightedRandomSampler(weight, 1))[0]
                        walker = neighbors.indices[idx]
                        neighbors = a_mat[walker]
                    if (i, r_idx) not in false_entity:
                        false_entity[(i, r_idx)] = []
                    false_entity[(i, r_idx)].append(walker)

    for entity, relation in false_entity:
        false_entity[(entity, relation)] = np.array(list(set(false_entity[(entity, relation)])))

    logging.info(f'randomly_sampled: {randomly_sampled}')

    np.save(file=save_path, arr=false_entity)

    # return false_entity

                        
                            

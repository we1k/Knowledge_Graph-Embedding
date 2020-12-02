import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma,
                double_entity_embedding=False, double_ralation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / self.hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a= -self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torc.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embbeding,
            a= -self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        if model_name == 'TransD':
            self.proj_entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
            nn.init.uniform_(
                tensor=proj_entity_embedding,
                a= -self.embedding_range.item(),
                b=self.embedding_range.item()
            )

            self.proj_relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
            nn.init.uniform_(
                tensor=self.proj_entity_embedding,
                a= -self.embedding_range.item(),
                b=self.embedding_range.item()
            )

        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        
        if model_name not in ['TransE', 'DisMult', 'ComplEx', 'RotatE', 'pRotatE', 'TransD']:
            raise ValueError('model {} not supported'.format(model_name))

        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use -- double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_ralation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_')
    
    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''
        if model == 'single':
            # sample.size() = (batch,3) [:,0] head采样, [:,1] relation采样 [:,2] tail
            batch_size, negative_sample_size = sample.size(0), 1
            # head.size() = (sample_size, 1,entity_dim) 
            head = torch.index_select(
                self.entity_embedding,  # (nentity, entity_dim)
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            # relation.size() = (sample_size, 1, relation_dim)
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:,1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:2]
            ).unsqueeze(1)

            if hasattr(self, 'proj_relation_embedding') and hasattr(self, 'proj_entity_embedding'):
                '''
                calculate the TransD
                '''
                pass
        
        elif mode == 'head-batch':
            tail_part, head_part = sample
            pass

        elif mode == 'tail-batch':
            pass

        else:
            raise ValueError('mode {} not supported'.format(mode))
        
        model_func = {
            'TransE': self.TransE,
            'TransD':
        }

def main():
    sample = torch.randint(0, 10, size=(4, 3)).long()
    print(sample)
    a = torch.randn(10, 4)
    print(a)
    b = torch.index_select(
        input=a,
        dim=0,
        index=sample[:,0]
    ).unsqueeze(1)
    print(b.size())

if __name__ == "__main__":
    main()
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
        
        # loss function logsigmoid(gamma - score)
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
        
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE', 'TransD']:
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
                head_t = torch.index_select(
                    self.proj_entity_embedding,
                    dim=0,
                    index=sample[:, 0]
                ).view(batch_size, negative_sample_size, -1)
                
                relation_t = torch.index_select(
                    self.proj_relation_embedding,
                    dim=0,
                    index=sample[:, 1]
                ).unsqueeze(1)

                tail_t = torch.index_select(
                    self.proj_relation_embedding,
                    dim=0,
                    index=sample[:, 2]
                ).unsqueeze(1)

            else:
                head_t = None
                relation_t = None
                tail_t = None
                           
        elif mode == 'head-batch':
            tail_part, head_part = sample
            # todo
            pass

        elif mode == 'tail-batch':
            # todo
            pass

        else:
            raise ValueError(f'mode {mode} not supported')
        
        model_func = {
            'TransE': self.TransE,
            'TransD': self.TransD,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, head_t, relation_t, tail_t, model)
        else:
            raise ValueError(f'model {self.model_name} not supported')

        return score
    def TransE(self, head, relation, tail, head_t, relation_t, tail_t, model):
        '''
        math:
            score = ||head, relation, tail||1  
            loss = gamma - ||score||1   
        '''
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        # gamma - N2 of score(dim 2 means the ) 
        # score.size()
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    # calculate the projection vector
    def _transfer(self, e, e_t, r_t):
        return F.normalize(e + (e * e_t).sum(dim=1, keepdim=True) * r_t, 2, -1)
    
    def TransD(self, head, relation, tail, head_t, relation_t, tail_t, model):
        '''
        math:
            score = ||head_proj, relation, tail_proj||1       
        '''
        head_proj = self._transfer(head, head_t, relation_t)
        tail_proj = self._transfer(tail, tail_t, relation_t)

        if mode == 'head-batch':
            score = head_proj + (relation - tail_proj)
        else:
            score = (head_proj + relation) - tail_proj
        
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score
        
    def DistMult(self, head, relation, tail, head_t, relation_t, tail_t, model):
        '''
        math:
            score = <head, relation, tail>        
        '''
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score
    
    def ComplEx(self, head, relation, tail, head_t, relation_t, tail_t, model):
        pass

    def RotatE(self, head, relation, tail, head_t, relation_t, tail_t, model):
        pass

    def pRotatE(self, head, relation, tail, head_t, relation_t, tail_t, model):
        pass

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step, apply back_propagation and return the loss
        '''

        # model(sample, model) = forward(sample, model)
        
        model.train()

        optimizer.zeros_grad()
        

        #  positive_sample, negative_sample size()?
        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        # negative sample size
        negative_score = model((positive_sample, negative_sample), mode=mode)

        # adv_RW
        # Loss = positive_score + adv * negative_score
        if args.negative_adversarial_sampling:
            negative_score = (F.softmax((negative_score * args) * args.adversarial_temperature, dim=1).detach()
                * F.logsigmoid(-negative_score)).sum(dim=1)

        else negative_score = F.logsigmoid(-negative_score).mean(dim=1)

        # single mode calculate positive_score
        positive_score = model(positive_sample)   

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        if args.uni_weight:
            positive_sample_loss = -positive_score.mean()
            negative_sample_loss = -negative_score.mean()
        else:
            positive_sample_loss = -(subsampling_weight * positive_score).sum() / subsampling_weight.sum()
            negative_sample_loss = -(subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2
        
        if args.regularization != 0.0:
            # Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_embedding.norm(p=3) ** 3 +
                model.relation_embbeding.norm(p=3) ** 3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log


    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        evaluate the model on the test
        '''
        model.eval()

        pass
        

def main():
    # sample = torch.randint(0, 10, size=(4, 3)).long()
    # print(sample)
    # a = torch.randn(10, 4)
    # print(a)
    # b = torch.index_select(
    #     input=a,
    #     dim=0,
    #     index=sample[:,0]
    # ).unsqueeze(1)
    # print(b.size())

    sample = torch.ones(3, 4)
    sample = sample.view(-1)
    print(sample)

if __name__ == "__main__":
    main()
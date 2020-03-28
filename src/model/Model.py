from tqdm import tqdm
import time
import os
import random
import torch
import numpy as np
from model.Test import testX
from model.Loss import scale_loss, orthogonal_loss


class Model():
    def __init__(self, config):
        self.epoch = config.epoch
        self.test_batch_size = config.test_batch_size
        self.early_stopping_round = config.early_stopping_round
        self.learning_rate = config.learning_rate
        self.norm = config.norm
        self.margin = config.margin
        self.C = config.C
        self.optimizer = config.optimizer
        self.loss_function = config.loss_function()
        self.entity_num = config.entity_num
        self.train_num = config.train_num
        self.train_data = config.train_data
        self.train_batch = config.train_batch
        self.valid_data = config.valid_data
        self.test_data = config.test_data
        self.model = config.model
        self.filename = config.filename
        self.processes = config.processes
        print('Model: {} | {}'.format(self.model.name, self.filename))

    def __triple_corrupted_head__(self, triple, train_data):
        while True:
            e = random.randrange(self.entity_num)
            if (e, triple[1], triple[2]) not in train_data:
                return (e, triple[1], triple[2])

    def __triple_corrupted_tail__(self, triple, train_data):
        while True:
            e = random.randrange(self.entity_num)
            if (triple[0], e, triple[2]) not in train_data:
                return (triple[0], e, triple[2])

    def __sample_corrupted_filter__(self, batch, train_data):
        corrupted = [self.__triple_corrupted_head__(triple, train_data) if random.random() < 0.5
                     else self.__triple_corrupted_tail__(triple, train_data)
                     for triple in batch]
        h = [triple[0] for triple in batch]
        t = [triple[1] for triple in batch]
        l = [triple[2] for triple in batch]
        h_apos = [triple[0] for triple in corrupted]
        t_apos = [triple[1] for triple in corrupted]
        l_apos = [triple[2] for triple in corrupted]
        return h, t, l, h_apos, t_apos, l_apos

    def train(self):
        model = self.model
        optimizer = self.optimizer(model.parameters(), lr=self.learning_rate)
        mean_rank = np.inf
        valid_epoch = 10
        mean_rank_not_decrease_count = 0
        lr_decrease = 0
        for epoch in range(self.epoch):
            print('Epoch {}'.format(epoch))
            start_time = time.time()
            loss = torch.FloatTensor([0.0])
            random.shuffle(self.train_batch)
            for batch in tqdm(self.train_batch):
                h, t, l, h_apos, t_apos, l_apos = \
                    self.__sample_corrupted_filter__(batch, self.train_data)
                h = torch.autograd.Variable(torch.LongTensor(h))
                t = torch.autograd.Variable(torch.LongTensor(t))
                l = torch.autograd.Variable(torch.LongTensor(l))
                h_apos = torch.autograd.Variable(torch.LongTensor(h_apos))
                t_apos = torch.autograd.Variable(torch.LongTensor(t_apos))
                l_apos = torch.autograd.Variable(torch.LongTensor(l_apos))

                model.zero_grad()
                dist, dist_apos = model(h, t, l,
                                        h_apos, t_apos, l_apos)

                batch_loss = self.loss_function(dist, dist_apos, self.margin)
                if model.name == 'TransH':
                    entity_embedding = model.entity_embedding(
                        torch.cat([h, t, h_apos, t_apos]))
                    relation_embedding = model.relation_embedding(
                        torch.cat([l, l_apos]))
                    w_embedding = model.w_embedding(
                        torch.cat([l, l_apos]))
                    batch_loss += self.C*(
                        scale_loss(entity_embedding) +
                        orthogonal_loss(relation_embedding, w_embedding)
                    )

                batch_loss.backward()
                optimizer.step()
                loss += batch_loss

            print("Epoch {} | loss: {} | {:.2f}s | lr: {}".format(
                epoch, loss.item(), time.time()-start_time, optimizer.param_groups[0]['lr']))

            if mean_rank < 300 and valid_epoch > 1:
                valid_epoch = 1
            elif mean_rank < 500 and valid_epoch > 5:
                valid_epoch = 5

            if epoch % valid_epoch == 0:
                entity_embedding = model.entity_embedding.weight.data.cpu().numpy()
                relation_embedding = model.relation_embedding.weight.data.cpu().numpy()
                w_embedding = None
                if model.name == 'TransH':
                    w_embedding = model.w_embedding.weight.data.cpu().numpy()
                valid_data = self.valid_data
                norm = self.norm
                processes = self.processes
                test_batch_size = self.test_batch_size
                _mean_rank, _ = testX(test_data=valid_data,
                                      entity_embedding=entity_embedding,
                                      relation_embedding=relation_embedding,
                                      w_embedding=w_embedding,
                                      norm=norm,
                                      model=model.name,
                                      processes=processes,
                                      batch_size=test_batch_size)
                if _mean_rank < mean_rank:
                    mean_rank_not_decrease_count = 0
                    print("Epoch {}: mean_rank improved from {:.2f} to {:.2f}, saving model".format(
                        epoch, mean_rank, _mean_rank))
                    mean_rank = _mean_rank
                    torch.save(model, os.path.join(
                        os.path.dirname(__file__), 'torch/'+model.name+'_'+self.filename+'.torch'))
                else:
                    print("Epoch {}: mean_rank {:.2f}".format(
                        epoch, _mean_rank))
                    mean_rank_not_decrease_count += 1
                    if mean_rank_not_decrease_count == 5:
                        lr_decrease += 1
                        if lr_decrease == self.early_stopping_round:
                            break
                        else:
                            optimizer.param_groups[0]['lr'] *= 0.1
                            mean_rank_not_decrease_count = 0

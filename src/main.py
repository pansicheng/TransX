from data.data import Data
from model.TransE import TransE
from model.TransH import TransH
from model.TransR import TransR
from model.Model import Model
from model.Loss import Loss
import torch


class Config():
    def __init__(self, path, TransX):
        self.epoch = 1000
        self.batch_size = 5000
        self.test_batch_size = 100
        self.early_stopping_round = 5
        self.learning_rate = 0.001
        self.e_dim = 50
        self.r_dim = 50
        self.norm = 2
        self.margin = 1
        self.C = 0.25
        self.optimizer = torch.optim.Adam
        self.loss_function = Loss
        data = Data(path, self.batch_size)
        self.entity_num = data.entity_num
        self.relation_num = data.relation_num
        self.train_num = data.train_num
        self.train_data = data.train_data
        self.train_batch = data.train_batch
        self.valid_data = list(data.valid_data.keys())
        self.test_data = list(data.test_data.keys())
        self.valid_data_classification = data.valid_data_classification
        self.test_data_classification = data.test_data_classification
        self.model = TransX(self.e_dim, self.r_dim, self.norm,
                            self.entity_num, self.relation_num,
                            path)
        self.filename = path
        self.processes = 4


def main():
    config = Config("WN18", TransE)
    model = Model(config)
    model.train()
    model.test()
    config = Config("FB15K", TransE)
    model = Model(config)
    model.train()
    model.test()
    config = Config("WN18", TransH)
    model = Model(config)
    model.train()
    model.test()
    config = Config("FB15K", TransH)
    model = Model(config)
    model.train()
    model.test()
    config = Config("WN18", TransR)
    model = Model(config)
    model.train()
    model.test()
    config = Config("FB15K", TransR)
    model = Model(config)
    model.train()
    model.test()


if __name__ == "__main__":
    main()

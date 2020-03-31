import random
import math
import os


class Data():
    def __init__(self, path, batch_size):
        self.entity_num, self.entity = self.__load_er__(os.path.join(
            os.path.dirname(__file__), "../../data/"+path+"/entity2id.txt"))
        self.relation_num, self.relation = self.__load_er__(os.path.join(
            os.path.dirname(__file__), "../../data/"+path+"/relation2id.txt"))
        self.train_num, self.train_data = self.__load_data__(os.path.join(
            os.path.dirname(__file__), "../../data/"+path+"/train2id.txt"))
        self.train_batch = self.__get_batch__(
            list(self.train_data.keys()), self.train_num, batch_size)
        _, self.valid_data = self.__load_data__(os.path.join(
            os.path.dirname(__file__), "../../data/"+path+"/valid2id.txt"))
        _, self.test_data = self.__load_data__(os.path.join(
            os.path.dirname(__file__), "../../data/"+path+"/test2id.txt"))
        _, self.r_left, self.r_right = self.__load_constrain__(os.path.join(
            os.path.dirname(__file__), "../../data/"+path+"/type_constrain.txt"))
        self.valid_data_classification = self.__classification__(
            list(self.valid_data.keys()), self.r_left, self.r_right)
        self.test_data_classification = self.__classification__(
            list(self.test_data.keys()), self.r_left, self.r_right)

    def __load_er__(self, path):
        with open(path, "r") as handle:
            num = int(handle.readline().strip())
            value2id = dict([
                (line.strip().split("\t")[0],
                 int(line.strip().split("\t")[1]))
                for line in handle.readlines()])
        return num, list(value2id.values())

    def __load_data__(self, path):
        with open(path, "r") as handle:
            num = int(handle.readline())
            data = dict([((int(line.strip().split()[0]),
                           int(line.strip().split()[1]),
                           int(line.strip().split()[2])), True)
                         for line in handle.readlines()])
        return num, data

    def __get_batch__(self, data, num, batch_size):
        batch = [data[i*batch_size:(i+1)*batch_size]
                 for i in range(math.ceil(num/batch_size))]
        return batch

    def __load_constrain__(self, path):
        r_left = {}
        r_right = {}
        with open(path, "r") as handle:
            num = int(handle.readline().strip())
            for _ in range(num):
                line = handle.readline().strip().split("\t")
                r_left[int(line[0])] = [int(v) for v in line[2:]]
                line = handle.readline().strip().split("\t")
                r_right[int(line[0])] = [int(v) for v in line[2:]]
        return num, r_left, r_right

    def __triple_corrupted_head__(self, triple, potential):
        if len(potential) == 1:
            potential = self.entity
        while True:
            e = random.sample(potential, 1)[0]
            if e != triple[0]:
                return (e, triple[1], triple[2])

    def __triple_corrupted_tail__(self, triple, potential):
        if len(potential) == 1:
            potential = self.entity
        while True:
            e = random.sample(potential, 1)[0]
            if e != triple[1]:
                return (triple[0], e, triple[2])

    def __classification__(self, test_data, r_left, r_right):
        # 根据 (Socher et al. 2013) corrupted 的数据要在 constrain 里面采样构造
        # (Pablo Picaso, nationality, Spain) -> (Pablo Picaso, nationality, United States)
        corrupted_test_data = [
            self.__triple_corrupted_head__(triple, r_left[triple[2]]) if random.random() < 0.5
            else self.__triple_corrupted_tail__(triple, r_right[triple[2]])
            for triple in test_data
        ]
        return dict([(v, True) for v in test_data]+[(v, False) for v in corrupted_test_data])

import math
import os


class Data():
    def __init__(self):
        pass

    def __load_er__(self, path):
        with open(path, 'r') as handle:
            num = int(handle.readline().strip())
            value2id = dict([
                (line.strip().split('\t')[0],
                 int(line.strip().split('\t')[1]))
                for line in handle.readlines()])
        return num, list(value2id.values())

    def __load_data__(self, path):
        with open(path, 'r') as handle:
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


class WN18(Data):
    def __init__(self, path, batch_size):
        self.entity_num, self.entity = self.__load_er__(os.path.join(
            os.path.dirname(__file__), '../../data/'+path+'/entity2id.txt'))
        self.relation_num, self.relation = self.__load_er__(os.path.join(
            os.path.dirname(__file__), '../../data/'+path+'/relation2id.txt'))
        self.train_num, self.train_data = self.__load_data__(os.path.join(
            os.path.dirname(__file__), '../../data/'+path+'/train2id.txt'))
        self.train_batch = self.__get_batch__(
            list(self.train_data.keys()), self.train_num, batch_size)
        _, self.valid_data = self.__load_data__(os.path.join(
            os.path.dirname(__file__), '../../data/'+path+'/valid2id.txt'))
        _, self.test_data = self.__load_data__(os.path.join(
            os.path.dirname(__file__), '../../data/'+path+'/test2id.txt'))

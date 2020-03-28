from tqdm import tqdm
import math
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from multiprocessing import Pool


def rank_where(h, t, hl, lt, norm, embedding):
    if norm == 1:
        dist_hl = pairwise_distances(
            hl, embedding, metric='manhattan')
        dist_lt = pairwise_distances(
            lt, embedding, metric='manhattan')
    else:
        dist_hl = pairwise_distances(
            hl, embedding, metric='euclidean')
        dist_lt = pairwise_distances(
            lt, embedding, metric='euclidean')
    t_rank = np.argsort(dist_hl, axis=1)
    h_rank = np.argsort(dist_lt, axis=1)
    return [np.where(line == _t)[0][0] for _t, line in zip(t, t_rank)] +\
        [np.where(line == _h)[0][0] for _h, line in zip(h, h_rank)]


def rank_E(test_data):
    global entity_embedding_global, relation_embedding_global, norm_global
    h = [triple[0] for triple in test_data]
    t = [triple[1] for triple in test_data]
    l = [triple[2] for triple in test_data]
    h_vec = entity_embedding_global[h]
    t_vec = entity_embedding_global[t]
    l_vec = relation_embedding_global[l]
    hl = h_vec+l_vec
    lt = t_vec-l_vec
    return rank_where(h, t, hl, lt, norm_global, entity_embedding_global)


def rank_H(test_data):
    global entity_embedding_global, relation_embedding_global, w_embedding_global, norm_global
    h = [triple[0] for triple in test_data]
    t = [triple[1] for triple in test_data]
    l = [triple[2] for triple in test_data]
    w_vec = w_embedding_global[l[0]]
    entity_embedding = entity_embedding_global - \
        np.sum(entity_embedding_global*w_vec, axis=1, keepdims=True)*w_vec
    h_vec = entity_embedding[h]
    t_vec = entity_embedding[t]
    l_vec = relation_embedding_global[l]
    hl = h_vec+l_vec
    lt = t_vec-l_vec
    return rank_where(h, t, hl, lt, norm_global, entity_embedding)


def testX(test_data, entity_embedding, relation_embedding, w_embedding,
          norm, model, processes, batch_size):
    global entity_embedding_global, relation_embedding_global, w_embedding_global, norm_global
    entity_embedding_global = entity_embedding
    relation_embedding_global = relation_embedding
    w_embedding_global = w_embedding
    norm_global = norm

    if model == 'TransE':
        rank_function = rank_E
        group = math.ceil(len(test_data)/batch_size)
        test_data = [test_data[i*batch_size:(i+1)*batch_size]
                     for i in range(group)]
    elif model == 'TransH':
        rank_function = rank_H
        _test_data = dict([(i, [])
                           for i in range(relation_embedding.shape[0])])
        # 每个 batch 只能有一种关系
        for triple in test_data:
            _test_data[triple[2]].append(triple)
        test_data = []
        # 每个 batch 的大小要求小于 batch_size
        for _test_sub in list(_test_data.values()):
            if (len(_test_sub) % batch_size) > (0.5*batch_size):
                for i in range(math.ceil(len(_test_sub)/batch_size)):
                    test_data.append(_test_sub[i*batch_size:(i+1)*batch_size])
            else:
                test_data.append(_test_sub)

    with Pool(processes) as pool:
        rank_list = [v for v in tqdm(
            pool.imap_unordered(rank_function, test_data), total=len(test_data)
        )]
    rank_list = [v for sublist in rank_list for v in sublist]
    mean_rank = sum(rank_list)/len(rank_list)
    hit10 = sum([1 for rank in rank_list if rank < 10])/len(rank_list)
    del entity_embedding_global, relation_embedding_global, norm_global
    return mean_rank, hit10

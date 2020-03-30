from tqdm import tqdm
import math
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from multiprocessing import Pool


# filter out those corrupted triples which have appeared in knowledge graph
def where_filter(h, t, l, line, head=True):
    global train_data_global
    where = 0
    for i in line:
        target, triple = (h, (i, t, l)) if head else (t, (h, i, l))
        if i == target:
            return where
        elif triple in train_data_global:
            continue
        else:
            where += 1
    return where


def rank_where(h, t, l, hl, lt, norm, embedding):
    global raw_global
    dist_hl, dist_lt = (
        pairwise_distances(hl, embedding, metric="manhattan"),
        pairwise_distances(lt, embedding, metric="manhattan")
    ) if norm == 1 else (
        pairwise_distances(hl, embedding, metric="euclidean"),
        pairwise_distances(lt, embedding, metric="euclidean")
    )
    t_rank = np.argsort(dist_hl, axis=1)
    h_rank = np.argsort(dist_lt, axis=1)
    rank_return = (
        [np.where(line == _t)[0][0] for _t, line in zip(t, t_rank)] +
        [np.where(line == _h)[0][0] for _h, line in zip(h, h_rank)]
    ) if raw_global else (
        [where_filter(_h, _t, _l, line, False)
         for _h, _t, _l, line in zip(h, t, l, t_rank)] +
        [where_filter(_h, _t, _l, line)
         for _h, _t, _l, line in zip(h, t, l, h_rank)]
    )
    return rank_return


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
    return rank_where(h, t, l, hl, lt, norm_global, entity_embedding_global)


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
    return rank_where(h, t, l, hl, lt, norm_global, entity_embedding)


def rank_R(test_data):
    global entity_embedding_global, relation_embedding_global, w_embedding_global, norm_global
    h = [triple[0] for triple in test_data]
    t = [triple[1] for triple in test_data]
    l = [triple[2] for triple in test_data]
    w_vec = w_embedding_global[l[0]].reshape((relation_embedding_global.shape[1],
                                              entity_embedding_global.shape[1])).T
    entity_embedding = np.matmul(entity_embedding_global, w_vec)
    h_vec = entity_embedding[h]
    t_vec = entity_embedding[t]
    l_vec = relation_embedding_global[l]
    hl = h_vec+l_vec
    lt = t_vec-l_vec
    return rank_where(h, t, l, hl, lt, norm_global, entity_embedding)


def relation_batch(test_data, relation_num, batch_size):
    _test_data = dict([(i, [])
                       for i in range(relation_num)])
    # 每个 batch 只能有一种关系
    for triple in test_data:
        _test_data[triple[2]].append(triple)
    test_data = []
    # 每个 batch 的大小要求不超过 batch_size
    for _test_sub in list(_test_data.values()):
        for i in range(math.ceil(len(_test_sub)/batch_size)):
            test_data.append(_test_sub[i*batch_size:(i+1)*batch_size])
    return test_data


def testX(test_data, train_data, entity_embedding, relation_embedding, w_embedding,
          norm, model, processes, batch_size, raw=True):
    global train_data_global, \
        entity_embedding_global, relation_embedding_global, w_embedding_global, \
        norm_global, raw_global
    train_data_global = train_data
    entity_embedding_global = entity_embedding
    relation_embedding_global = relation_embedding
    w_embedding_global = w_embedding
    norm_global = norm
    raw_global = raw

    if model == "TransE":
        rank_function = rank_E
        group = math.ceil(len(test_data)/batch_size)
        test_data = [test_data[i*batch_size:(i+1)*batch_size]
                     for i in range(group)]
    elif model == "TransH":
        rank_function = rank_H
        test_data = relation_batch(
            test_data, relation_embedding.shape[0], batch_size)
    elif model == "TransR":
        rank_function = rank_R
        test_data = relation_batch(
            test_data, relation_embedding.shape[0], batch_size)

    desc_format = "raw" if raw else "filter"
    with Pool(processes) as pool:
        rank_list = [v for v in tqdm(
            pool.imap_unordered(rank_function, test_data),
            desc="link prediction {:7}".format(desc_format),
            total=len(test_data)
        )]
    rank_list = [v for sublist in rank_list for v in sublist]
    mean_rank = sum(rank_list)/len(rank_list)
    hit10 = sum([1 for rank in rank_list if rank < 10])/len(rank_list)
    del train_data_global, \
        entity_embedding_global, relation_embedding_global, w_embedding_global, \
        norm_global, raw_global
    return int(mean_rank), hit10


# valid_data, test_data: {(h, t, l): True}
def testC(valid_data, test_data, entity_embedding, relation_embedding, w_embedding, norm, model):
    # _valid_data, _test_data: {0: [(h, t, l), ]}
    _valid_data = dict([(i, []) for i in range(relation_embedding.shape[0])])
    for triple in list(valid_data.keys()):
        _valid_data[triple[2]].append(triple)
    _test_data = dict([(i, []) for i in range(relation_embedding.shape[0])])
    for triple in list(test_data.keys()):
        _test_data[triple[2]].append(triple)

    test_tp, test_fp, test_tn, test_fn = 0, 0, 0, 0
    for relation in tqdm(range(relation_embedding.shape[0]),
                         desc="triple classification  "):
        if model == "TransE":
            _entity_embedding = entity_embedding
        elif model == "TransH":
            w_vec = w_embedding[relation]
            _entity_embedding = entity_embedding - \
                np.sum(entity_embedding*w_vec, axis=1, keepdims=True)*w_vec
        elif model == "TransR":
            w_vec = w_embedding[relation].reshape((relation_embedding.shape[1],
                                                   entity_embedding.shape[1])).T
            _entity_embedding = np.matmul(entity_embedding, w_vec)

        # __valid_data, __test_data: [(h, t, l), ]
        __valid_data = _valid_data[relation]
        h = [triple[0] for triple in __valid_data]
        t = [triple[1] for triple in __valid_data]
        l = [triple[2] for triple in __valid_data]
        h_vec = _entity_embedding[h]
        t_vec = _entity_embedding[t]
        l_vec = relation_embedding[l]
        dist = np.linalg.norm(h_vec+l_vec-t_vec, ord=norm, axis=1)
        dist_rank = np.argsort(dist)
        # positive 和 negative 数据各一半
        tp, fp, tn, fn = 0, 0, len(dist)*0.5, len(dist)*0.5
        acc = (tp+tn)/(tp+fp+tn+fn)
        # sigma 为阈值， dist <= sigma 预测为真
        sigma = dist[dist_rank[0]]
        for i, v in enumerate(dist_rank):
            tp, fp, tn, fn = (tp+1, fp, tn, fn-1) \
                if valid_data[__valid_data[v]] else \
                (tp, fp+1, tn-1, fn)
            _acc = (tp+tn)/(tp+fp+tn+fn)
            if _acc > acc:
                acc = _acc
                sigma = (dist[v]+dist[dist_rank[i+1]])/2 \
                    if i+1 < len(dist_rank) else \
                    dist[v]-np.finfo(dist[v].dtype).eps
        tp, fp, tn, fn = 0, 0, 0, 0
        __test_data = _test_data[relation]
        h = [triple[0] for triple in __test_data]
        t = [triple[1] for triple in __test_data]
        l = [triple[2] for triple in __test_data]
        h_vec = _entity_embedding[h]
        t_vec = _entity_embedding[t]
        l_vec = relation_embedding[l]
        dist = np.linalg.norm(h_vec+l_vec-t_vec, ord=norm, axis=1)
        for i, dist_i in enumerate(dist):
            pred = dist_i <= sigma
            if test_data[__test_data[i]]:
                if pred:
                    tp += 1
                else:
                    fp += 1
            else:
                if pred:
                    fn += 1
                else:
                    tn += 1
        test_tp += tp
        test_fp += fp
        test_tn += tn
        test_fn += fn

    return (tp+tn)/(tp+fp+tn+fn)

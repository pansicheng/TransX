import os
import torch
from model.TransE import TransE


class TransR(torch.nn.Module):
    def __init__(self,
                 e_dim, r_dim, norm,
                 entity_num,
                 relation_num,
                 path
                 ):
        super(TransR, self).__init__()
        self.name = "TransR"
        self.dim = e_dim
        self.norm = norm
        self.entity_num = entity_num
        self.relation_num = relation_num

        # initialize entity and relation
        # embeddings with results of TransE
        modelE = TransE(e_dim, e_dim, norm,
                        entity_num, relation_num,
                        path)
        modelE = torch.load(os.path.join(
            os.path.dirname(__file__),
            "torch/TransE"+"_"+path+".torch"))
        entity_embeddingE = modelE.entity_embedding.weight.data
        relation_embeddingE = modelE.relation_embedding.weight.data
        self.entity_embedding = torch.nn.Embedding(
            self.entity_num, self.dim)
        self.relation_embedding = torch.nn.Embedding(
            self.relation_num, self.dim)
        self.w_embedding = torch.nn.Embedding(
            self.relation_num, self.dim**2)
        self.entity_embedding.weight.data = entity_embeddingE
        self.relation_embedding.weight.data = relation_embeddingE

        # and initialize relation matrices
        # as identity matrices
        init_w_embedding_weight = torch.nn.init.eye_(
            torch.FloatTensor(self.dim, self.dim)).view(-1, self.dim**2)
        init_w_embedding_weight = torch.cat([init_w_embedding_weight
                                             for _ in range(self.relation_num)])
        self.w_embedding.weight.data = init_w_embedding_weight

    def forward(self, h_batch, t_batch, l_batch, h_apos_batch, t_apos_batch, l_apos_batch):
        h_vec = self.entity_embedding(h_batch).view(-1, self.dim, 1)
        l_vec = self.relation_embedding(l_batch)
        t_vec = self.entity_embedding(t_batch).view(-1, self.dim, 1)
        w_vec = self.w_embedding(
            l_batch).view(-1, self.dim, self.dim)

        h_apos_vec = self.entity_embedding(h_apos_batch).view(-1, self.dim, 1)
        l_apos_vec = self.relation_embedding(l_apos_batch)
        t_apos_vec = self.entity_embedding(t_apos_batch).view(-1, self.dim, 1)
        w_apos_vec = self.w_embedding(
            l_apos_batch).view(-1, self.dim, self.dim)

        h_perp = torch.matmul(w_vec, h_vec).view(-1, self.dim)
        t_perp = torch.matmul(w_vec, t_vec).view(-1, self.dim)
        h_apos_perp = torch.matmul(w_apos_vec, h_apos_vec).view(-1, self.dim)
        t_apos_perp = torch.matmul(w_apos_vec, t_apos_vec).view(-1, self.dim)

        dist = torch.norm(h_perp + l_vec - t_perp,
                          self.norm, 1)
        dist_apos = torch.norm(h_apos_perp + l_apos_vec - t_apos_perp,
                               self.norm, 1)

        return dist, dist_apos, h_perp, t_perp, h_apos_perp, t_apos_perp

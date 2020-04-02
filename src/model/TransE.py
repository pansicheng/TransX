import torch


class TransE(torch.nn.Module):
    def __init__(self,
                 e_dim, r_dim, norm,
                 entity_num,
                 relation_num,
                 path
                 ):
        super(TransE, self).__init__()
        self.name = "TransE"
        self.dim = e_dim
        self.norm = norm
        self.entity_num = entity_num
        self.relation_num = relation_num
        _tmp = 6/(self.dim**0.5)

        self.entity_embedding = torch.nn.Embedding(
            self.entity_num, self.dim)
        self.relation_embedding = torch.nn.Embedding(
            self.relation_num, self.dim)
        torch.nn.init.uniform_(self.entity_embedding.weight, -_tmp, _tmp)
        torch.nn.init.uniform_(self.relation_embedding.weight, -_tmp, _tmp)
        normalize_entity_embedding = torch.nn.functional.normalize(
            self.entity_embedding.weight.data, p=2, dim=1)
        normalize_relation_embedding = torch.nn.functional.normalize(
            self.relation_embedding.weight.data, p=2, dim=1)
        self.entity_embedding.weight.data = normalize_entity_embedding
        self.relation_embedding.weight.data = normalize_relation_embedding

    def forward(self, h_batch, t_batch, l_batch, h_apos_batch, t_apos_batch, l_apos_batch):
        h_vec = self.entity_embedding(h_batch)
        l_vec = self.relation_embedding(l_batch)
        t_vec = self.entity_embedding(t_batch)
        h_apos_vec = self.entity_embedding(h_apos_batch)
        l_apos_vec = self.relation_embedding(l_apos_batch)
        t_apos_vec = self.entity_embedding(t_apos_batch)

        dist = torch.norm(h_vec + l_vec - t_vec,
                          self.norm, 1)
        dist_apos = torch.norm(h_apos_vec + l_apos_vec - t_apos_vec,
                               self.norm, 1)

        return dist, dist_apos, h_vec, t_vec, h_apos_vec, t_apos_vec

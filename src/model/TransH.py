import torch


class TransH(torch.nn.Module):
    def __init__(self,
                 e_dim, r_dim, norm,
                 entity_num,
                 relation_num,
                 path
                 ):
        super(TransH, self).__init__()
        self.name = "TransH"
        self.dim = e_dim
        self.norm = norm
        self.entity_num = entity_num
        self.relation_num = relation_num

        self.entity_embedding = torch.nn.Embedding(
            self.entity_num, self.dim)
        self.relation_embedding = torch.nn.Embedding(
            self.relation_num, self.dim)
        self.w_embedding = torch.nn.Embedding(
            self.relation_num, self.dim)
        torch.nn.init.xavier_normal_(self.entity_embedding.weight)
        torch.nn.init.xavier_normal_(self.relation_embedding.weight)
        torch.nn.init.xavier_normal_(self.w_embedding.weight)
        normalize_entity_embedding = torch.nn.functional.normalize(
            self.entity_embedding.weight.data, p=2, dim=1)
        normalize_relation_embedding = torch.nn.functional.normalize(
            self.relation_embedding.weight.data, p=2, dim=1)
        normalize_w_embedding = torch.nn.functional.normalize(
            self.w_embedding.weight.data, p=2, dim=1)
        self.entity_embedding.weight.data = normalize_entity_embedding
        self.relation_embedding.weight.data = normalize_relation_embedding
        self.w_embedding.weight.data = normalize_w_embedding

    def forward(self, h_batch, t_batch, l_batch, h_apos_batch, t_apos_batch, l_apos_batch):
        normalize_w_embedding = torch.nn.functional.normalize(
            self.w_embedding.weight.data, p=2, dim=1)
        self.w_embedding.weight.data = normalize_w_embedding

        h_vec = self.entity_embedding(h_batch)
        l_vec = self.relation_embedding(l_batch)
        t_vec = self.entity_embedding(t_batch)
        w_vec = self.w_embedding(l_batch)
        h_apos_vec = self.entity_embedding(h_apos_batch)
        l_apos_vec = self.relation_embedding(l_apos_batch)
        t_apos_vec = self.entity_embedding(t_apos_batch)
        w_apos_vec = self.w_embedding(l_apos_batch)

        h_perp = h_vec-torch.sum(h_vec*w_vec, dim=1, keepdim=True)*w_vec
        t_perp = t_vec-torch.sum(t_vec*w_vec, dim=1, keepdim=True)*w_vec
        h_apos_perp = h_apos_vec - \
            torch.sum(h_apos_vec*w_apos_vec, dim=1, keepdim=True)*w_apos_vec
        t_apos_perp = t_apos_vec - \
            torch.sum(t_apos_vec*w_apos_vec, dim=1, keepdim=True)*w_apos_vec

        dist = torch.norm(h_perp + l_vec - t_perp,
                          self.norm, 1)
        dist_apos = torch.norm(h_apos_perp + l_apos_vec - t_apos_perp,
                               self.norm, 1)

        return dist, dist_apos, h_perp, t_perp, h_apos_perp, t_apos_perp

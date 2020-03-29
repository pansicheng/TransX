import torch


class TransR(torch.nn.Module):
    def __init__(self,
                 e_dim, r_dim, norm,
                 entity_num,
                 relation_num
                 ):
        """Training set S = {(h, l, t)}, entities and rel. sets E and L, margin gamma, embeddings dim. k."""
        super(TransR, self).__init__()
        self.name = 'TransR'
        self.e_dim = e_dim
        self.r_dim = r_dim
        self.norm = norm
        self.entity_num = entity_num
        self.relation_num = relation_num

        self.entity_embedding = torch.nn.Embedding(
            self.entity_num, self.e_dim)
        self.relation_embedding = torch.nn.Embedding(
            self.relation_num, self.r_dim)
        self.w_embedding = torch.nn.Embedding(
            self.relation_num, self.r_dim*self.e_dim)
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
        normalize_entity_embedding = torch.nn.functional.normalize(
            self.entity_embedding.weight.data, p=2, dim=1)
        normalize_relation_embedding = torch.nn.functional.normalize(
            self.relation_embedding.weight.data, p=2, dim=1)
        normalize_w_embedding = torch.nn.functional.normalize(
            self.w_embedding.weight.data, p=2, dim=1)
        self.entity_embedding.weight.data = normalize_entity_embedding
        self.relation_embedding.weight.data = normalize_relation_embedding
        self.w_embedding.weight.data = normalize_w_embedding

        h_vec = self.entity_embedding(h_batch).view(-1, self.e_dim, 1)
        l_vec = self.relation_embedding(l_batch)
        t_vec = self.entity_embedding(t_batch).view(-1, self.e_dim, 1)
        w_vec = self.w_embedding(l_batch).view(-1, self.r_dim, self.e_dim)
        h_apos_vec = self.entity_embedding(
            h_apos_batch).view(-1, self.e_dim, 1)
        l_apos_vec = self.relation_embedding(l_apos_batch)
        t_apos_vec = self.entity_embedding(
            t_apos_batch).view(-1, self.e_dim, 1)
        w_apos_vec = self.w_embedding(
            l_apos_batch).view(-1, self.r_dim, self.e_dim)

        h_perp = torch.matmul(w_vec, h_vec).view(-1, self.r_dim)
        t_perp = torch.matmul(w_vec, t_vec).view(-1, self.r_dim)
        h_apos_perp = torch.matmul(w_apos_vec, h_apos_vec).view(-1, self.r_dim)
        t_apos_perp = torch.matmul(w_apos_vec, t_apos_vec).view(-1, self.r_dim)

        dist = torch.norm(h_perp + l_vec - t_perp,
                          self.norm, 1)
        dist_apos = torch.norm(h_apos_perp + l_apos_vec - t_apos_perp,
                               self.norm, 1)

        return dist, dist_apos

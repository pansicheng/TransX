import torch


class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, dist, dist_apos, margin):
        zero_tensor = torch.autograd.Variable(
            torch.FloatTensor(dist.size()).zero_()
        )
        return torch.sum(torch.max(dist - dist_apos + margin, zero_tensor))


def scale_loss(embedding):
    return torch.sum(
        torch.max(
            torch.sum(
                embedding ** 2, dim=1, keepdim=True
            )-torch.autograd.Variable(torch.FloatTensor([1.0])),
            torch.autograd.Variable(torch.FloatTensor([0.0]))
        ))


def orthogonal_loss(relation_embedding, w_embedding):
    return torch.sum(
        torch.sum(
            relation_embedding * w_embedding, dim=1, keepdim=True
        ) ** 2 / torch.sum(relation_embedding ** 2, dim=1, keepdim=True)
    )

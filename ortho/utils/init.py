from .. import torch


def haar_orthogonal(weight : torch.Tensor):
    assert weight.dim() == 2, f'Dimension must be 2, got {weight.dim()}.'
    m, n = weight.shape
    with torch.no_grad():
        x = torch.randn_like(weight)

        if m < n:
            x = x.T

        Q, R = torch.linalg.qr(x)
        d = torch.diag(R).sign()
        Q *= d.unsqueeze(-2).expand_as(Q)

        if m < n:
            Q = Q.T

        if m == n:
            mask = (torch.det(Q) > 0.0).float()
            mask[mask == 0.0] = -1.0
            mask = mask.unsqueeze(-1).unsqueeze(-1).expand_as(Q)
            Q[..., 0] *= mask[..., 0]

    return Q

def haar_orthogonal_(weight : torch.Tensor):
    weight.copy_(haar_orthogonal(weight))
    return weight
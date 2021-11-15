from .. import (
    torch, Union, F, np
)

_reductions = {
    'mean': torch.mean,
    'sum': torch.sum,
    'none': lambda x: x
}


def logcosh(input: torch.Tensor, target: torch.Tensor, alpha: float = 1., eps: float = 1e-7, reduction: str = 'mean'):
    """Log-Hyperbolic Cosine loss.

    NOTE: The to make the function more robust, we compute a threshold where the gradients are less than epsilon.
    """
    # pylint: disable=no-member
    redfunc = _reductions[reduction.lower()]
    errval = np.arctanh(1 - eps) / alpha
    c = errval - np.log(np.cosh(alpha * errval)) / alpha
    abs = F.l1_loss(input, target, reduction='none')
    out = (abs > errval) * (abs - c)
    out[abs <= errval] = torch.log(torch.cosh(alpha * (input - target)[abs <= errval])) / alpha
    return redfunc(out)

def norm(input: torch.Tensor, target: torch.Tensor, p: Union[int, str] = None, reduction: str = 'mean'):
    """Norm loss function.
    """
    # pylint: disable=no-member
    redfunc = _reductions[reduction.lower()]
    return redfunc(torch.linalg.norm(input - target, ord=p, dim=-1))

def kl_divergence(mu: torch.Tensor, lv: torch.Tensor, reduction: str = 'sum', total_size : int = None):
    """Kullback-Leibler divergence for regularization of Gaussian Variational Encoder-Decoder.
    """
    redfunc = _reductions[reduction.lower()]
    weight = -.5
    if reduction.lower() == 'mean':
        assert total_size is not None, 'KLD requires total_size of dataset for mean reduction.'
        weight *= len(mu) / total_size

    return weight * redfunc(1 + lv - mu.pow(2) - lv.exp())

def equiangular_loss(M, ord=torch.inf):
    '''Regularization to promote equiangular properties.
    '''

    if M.shape[0] < M.shape[1]:
        m, n = M.shape
        G = M.T @ M
    else:
        n, m = M.shape
        G = M @ M.T

    I = torch.eye(n).to(G.device)
    v = (n - m)/((n-1)*m)
    E = (torch.ones_like(G) - I) * v + I

    #return torch.max(torch.abs(G - E))
    return torch.linalg.norm(torch.abs(G - E), ord=ord)
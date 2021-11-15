from .. import (
    torch
)

def project_into_ball(r, radius=1, ord=2):
    ''' Projects a vector onto a ball in L^p space.

    Parameters
    ----------
    r : torch.tensor
        A perturbation vector

    radius : float, optional
        The maximum radius of the vector.

    ord : int or str, optional
        Order of norm to use for projection.

    Returns
    -------
    A perturbation vector projected inside a ball with given radius.
    '''
    if r.dim() == 2:
        return radius * r / torch.linalg.norm(r, ord=ord)
        # norm = torch.linalg.norm(r, ord=ord, dim=-1) / radius
        # outside = torch.einsum('np,n->np', r, (norm > 1) / norm)
        # inside = torch.einsum('np,n->np', r, norm <= 1)
        # return outside + inside

    norm = torch.linalg.norm(r, ord=ord) / radius
    return r / norm if norm > 1 else r


@torch.jit.script
def do_digit(x: torch.Tensor, n: int, base: int=2, dim: int=-1) -> torch.Tensor:
    length = x.shape[-1]
    pos = (n+1) * length
    digits = torch.div(x, base**n, rounding_mode='trunc') % base
    exprange = torch.arange(start=pos, end=pos-length, step=-1).float().to(x.device)
    return torch.sum(base**(exprange) * digits, dim=dim)

@torch.jit.script
def do_digits(x: torch.Tensor, iter: int=27, base: int=2, dim: int=-1) -> torch.Tensor:
    out = torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)
    for n in range(-1, -iter, -1):
        out += do_digit(x, n, base=base, dim=dim)
    return out / base

@torch.jit.script
def undo_digit(x: torch.Tensor, n: int, length: int, base: int=2) -> torch.Tensor:
    shape = list(x.shape) + [length]
    z = torch.zeros(shape, dtype=x.dtype, device=x.device)
    for k in range(length):
        h = torch.div(x, base**(n-k+1), rounding_mode='trunc') % base
        z[...,k] = h
    return z

@torch.jit.script
def undo_digits(x: torch.Tensor, length: int, iter: int=27, base: int=2) -> torch.Tensor:
    shape = list(x.shape) + [length]
    out = torch.zeros(shape, dtype=x.dtype, device=x.device)
    x = x * base
    for k, n in enumerate(range(-1, -iter*length, -length)):
        z = undo_digit(x, n, length, base=base)
        if out is None:
            out = z * base**(-k-1)
        else:
            out += z * base**(-k-1)
    return out
from .. import torch

from . import (
    metrics, gpu, rng, projection, loss, reporting, transforms, init, convutils
)

from .dct import (
    dct, dct2, idct, idct2
)

def matrix_fracpow(A, p):
    val, vec = torch.linalg.eig(A)
    return vec @ torch.diag(val**(p)) @ torch.inverse(vec)

def reparametrize_gaussian(mu, lv) -> torch.Tensor:
    '''Reparametrization Trick for Gaussian VAE.

    Parameters
    ----------
    mu : torch.tensor
        Mean vector.

    lv : torch.tensor
        Log Variance vector.

    Returns
    -------
    A sample from a multinormal distribution with diagonal covariance matrix.
    '''
    std = torch.exp(0.5 * lv)
    eps = torch.randn_like(mu)
    return mu + (eps * std)


def winsplit(tensor, split):
    '''Function to split images into windows for use in dense seperable mixer layers.

    Note: The channel dimension C is optional.

    Parameters
    ----------
    tensor: torch.tensor
        A tensor with a batch of B, C-channel images on the form BxCxHxW or BxHxW.

    split: tuple(int, int)
        A tuple of ints giving the size of the windows for the split.
        For a batch of BxCx384x384, a split of (24, 24) will split the images into
        (384 / 24)^2 = (16)^2 = 256 patches of size (24, 24).

    Returns
    -------
    An output tensor of window patches with dimension BxCxPxS1xS2.
    '''
    n = tensor.dim()
    if n == 4:
        perm1 = (0,1,2,4,3,5)
    elif n == 3:
        perm1 = (0,1,3,2,4)
    else:
        raise ValueError(f'Dimension {n} not supported!')

    a = torch.stack(torch.split(tensor, split[0], dim=n-2), dim=n-2)
    b = torch.stack(torch.split(a, split[1], dim=-1), dim=n)
    c = b.permute(*perm1).flatten(n-2,n-1)
    return c


def winunsplit(tensor, grid):
    '''Function to reverse window splitting for dense seperable mixer layers.

    Note: The channel dimension C is optional.

    Parameters
    ----------
    tensor: torch.tensor
        A tensor with a batch of B, C-channel images on the form BxCxHxW or BxHxW.

    grid: tuple(int, int)
        Tuple of ints which gives the unflattened dimension of the split grid from the
        winsplit function. Given the same example of a batch of BxCx384x384 images, we
        perform the winsplit function, yielding an output of BxCx256x24x24. The number
        of output patches is from splitting the images into a grid of
        (384/24)x(384/24)=16x16=256 patches. To restore the image, the grid dimensions
        should be provided.

    Returns
    -------
    A tensor with a batch of images of dimension BxCxWxH.
    '''
    m = tensor.dim()
    if m == 5:
        perm1 = (0,1,2,4,3,5)
        perm2 = (2,3,4,5,0,1)
        perm3 = (1,2,3,4,0)
        perm4 = (1,2,3,0)
    elif m == 4:
        perm1 = (0,1,3,2,4)
        perm2 = (1,2,3,4,0)
        perm3 = (1,2,3,0)
        perm4 = (1,2,0)
    else:
        raise ValueError(f'Dimension {m} not supported!')

    d = tensor.unflatten(m-3, grid).permute(*perm1)
    e = torch.cat(tuple(d.permute(*perm2)))
    f = torch.cat(tuple(e.permute(*perm3)))
    g = f.permute(*perm4)

    return g


def generate_skew_indices(n : int):
    i_a = torch.hstack([
        torch.vstack([
                torch.arange(k),
                torch.arange(k) + n - k
            ])
        for k in torch.arange(n-1,0,-1)
    ])

    i_b = torch.hstack([
        torch.vstack([
                torch.arange(k) + n - k,
                torch.arange(k)
            ])
        for k in torch.arange(n-1,0,-1)
    ])
    return torch.hstack([i_a, i_b])


def default_dictonary_getter(actual, **default):
    return {
        k:(
            actual.get(k)
                if k in actual
                else default.get(k)
        ) for k in default.keys()
    }
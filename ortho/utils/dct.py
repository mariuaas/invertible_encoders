from .. import (
    torch, np, math, fft
)

def dct(x: torch.Tensor, norm: str = 'ortho') -> torch.Tensor:
    '''Discrete Cosine transform.

    Discrete Cosine Transform Type II (DCT-II) adapted for Pytorch. Uses a mirrored 2N
    representation, described by Makhoul (https://ieeexplore.ieee.org/document/1163351).

    Parameters
    ----------
    x: torch.Tensor
        Input tensor.

    norm: str
        Normalization, either 'ortho' or None.

    Returns
    -------
    Tensor where the one dimensional DCT-II applied to the last dimension.
    '''
    N = x.shape[-1]
    u = torch.zeros(*x.shape[:-1], 2*N)
    u[...,:N] = x[...,:]
    u[...,N:] = x[...,:].flip(-1)
    U = fft.fft(u)[...,:N]
    k = torch.arange(N)/(2*N)
    U[...,:] *= torch.exp(-1j*np.pi*k)

    if norm == 'ortho':
        U[...,0] *= math.sqrt(1/N) / 2
        U[...,1:] *= math.sqrt(2/N) / 2
    return U.real


def idct(y: torch.Tensor, norm: str = 'ortho') -> torch.Tensor:
    '''Inverse Discrete Cosine transform.

    Inverse Discrete Cosine Transform (DCT-III) adapted for Pytorch. Uses a mirrored 2N
    representation, described by Makhoul (https://ieeexplore.ieee.org/document/1163351).

    Parameters
    ----------
    y: torch.Tensor
        Input tensor.

    norm: str
        Normalization, either 'ortho' or None.

    Returns
    -------
    Tensor where the one dimensional DCT-III applied to the last dimension.
    '''
    N = y.shape[-1]
    y = y.clone()
    if norm == 'ortho':
        y[...,0] /= math.sqrt(1/N) / 2
        y[...,1:] /= math.sqrt(2/N) / 2
    k = torch.arange(N)/(2*N)
    Q = torch.exp(-np.pi*1j * k)
    u = torch.zeros_like(y, dtype=torch.cfloat)
    u[...,0] = y[...,0]/(2*Q[0])
    u[...,1:] = (y[...,1:] - 1j*y.flip(-1)[...,:-1])/(2*Q[1:])
    u = fft.ifft(u)
    U = torch.zeros_like(y)
    U[...,0::2] = u.real[...,:math.ceil(N/2)]
    U[...,1::2] = u.real.flip(-1)[...,:int(N/2)]
    return U


def dct2(x: torch.Tensor, norm: str = 'ortho') -> torch.Tensor:
    '''Two Dimensional Discrete Cosine transform.

    2D version of Discrete Cosine Transform (DCT-II) adapted for Pytorch.

    Parameters
    ----------
    x: torch.Tensor
        Input tensor.

    norm: str
        Normalization, either 'ortho' or None.

    Returns
    -------
    Tensor where the one dimensional DCT-III applied to the last dimension.
    '''
    return dct(
        dct(x, norm=norm).transpose(-1, -2),
        norm=norm
    ).transpose(-1, -2)


def idct2(y: torch.Tensor, norm: str = 'ortho') -> torch.Tensor:
    '''Two Dimensional Inverse Discrete Cosine transform.

    2D version of Inverse Discrete Cosine Transform (DCT-III) adapted for Pytorch.

    Parameters
    ----------
    y: torch.Tensor
        Input tensor.

    norm: str
        Normalization, either 'ortho' or None.

    Returns
    -------
    Tensor where the one dimensional DCT-III applied to the last dimension.
    '''
    return idct(
        idct(y, norm=norm).transpose(-1, -2),
        norm=norm
    ).transpose(-1, -2)

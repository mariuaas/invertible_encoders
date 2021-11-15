from .. import (
    np, nn, torch, F, fft, OrderedDict, Union
)

from .ssim import ssim, SSIM


def psnr(actual : torch.Tensor, approx: torch.Tensor, scale: float = 1.0):
    '''Calculates peak signal-to-noise ratio between two images.

    Parameters
    ----------
    actual: torch.Tensor
        Reference image as tensor
    approx: torch.Tensor
        Approximated image as tensor
    '''
    mse = F.mse_loss(actual, approx)
    return 20 * torch.log10(scale / torch.sqrt(mse))


def stochastic_spectral_norm(W: torch.Tensor, iterations: int) -> torch.Tensor:
    '''Estimates the spectral norm of a matrix.

    Calculates a differentiable stochastic estimation of the largest singular value of a
    matrix, which can be used for regularization.

    Ref: Yoshida, Miyato 2017
    '''
    _, n = W.shape
    device = W.device
    dtype = torch.cfloat if W.is_complex() else torch.float
    v = torch.randn(n, dtype=dtype).to(device)
    for _ in range(iterations):
        u = W @ v
        if torch.is_complex(W):
            v = W.t().conj() @ u
        else:
            v = W.t() @ u

    return torch.linalg.norm(v) / torch.linalg.norm(u)


def hilbert_schmidt_norm(W: torch.Tensor) -> torch.Tensor:
    """Calculates the Hilbert-Schmidt norm of a matrix W.

    The Hilbert-Schmidt norm is given by trace(W * W^T).
    """
    return torch.trace(W.T @ W)


def relative_error(
        actual: torch.Tensor, approx: torch.Tensor, p: Union[int, str] = 2,
        reduction: str = 'mean', disregard_zeros: bool = True
    ) -> torch.Tensor:
    '''Computes the relative error of a vector approximation.

    Parameters
    ----------
    actual : torch.tensor
        The reference vector.

    approx : torch.tensor
        The approximation vector.

    p : int, optional
        The norm over which to compute relative error.

    reduction : str, optional
        Reduction type, either 'mean', 'median', or 'sum'.

    Returns
    -------
    The relative error of the approximation.
    '''
    x0 = actual.detach().to("cpu")
    x = approx.detach().to("cpu")

    redfunc = {
        'mean': torch.mean,
        'sum': torch.sum,
        'median': torch.median,
    }[reduction]

    norm_e = torch.linalg.norm(x0 - x, ord=p, dim=-1)
    norm_r = torch.linalg.norm(x0, ord=p, dim=-1)

    if disregard_zeros:
        nonzeros = norm_r != 0
        norm_r = norm_r[nonzeros]
        norm_e = norm_e[nonzeros]

    return redfunc(norm_e / norm_r)


def relative_cond_no(
        factual: torch.Tensor, fapprox: torch.Tensor, x0: float, delta: float,
        p: Union[int, str] = 2, reduction: str = "mean", disregard_zeros: bool = True
    ):
    """Computes relative condition number of observations from alternate mappings.
    """

    cond_f = relative_error(
        factual, fapprox, p=p, reduction=reduction, disregard_zeros=disregard_zeros
    )

    redfunc = {
        'mean': torch.mean,
        'sum': torch.sum,
        'median': torch.median,
    }[reduction]

    norm_x0 = redfunc(torch.linalg.norm(x0, ord=p, dim=0))
    norm_delta = redfunc(torch.linalg.norm(delta, ord=p, dim=0))

    if disregard_zeros:
        nonzeros = norm_x0 != 0
        norm_x0 = norm_x0[nonzeros]
        norm_delta = norm_delta[nonzeros]

    cond_x = norm_delta / norm_x0

    # NOTE: Return might fail when disregarding zeros?

    return cond_f / cond_x


def relative_error_matrix(actual, approx, p=None, dim=None):
    '''Computes the relative error of a matrix approximation.

    Parameters
    ----------
    actual : torch.tensor
        The reference vector.

    approx : torch.tensor
        The approximation vector.

    p : int, optional
        The norm over which to compute relative error.

    Returns
    -------
    The relative error of a matrix approximation.
    '''
    Ah = approx.detach().to("cpu")
    A = actual.detach().to("cpu")

    if p != 'hs':
        norm_e = torch.linalg.norm(Ah - A, ord=p, dim=dim)
        norm_r = torch.linalg.norm(A, ord=p, dim=dim)

    else:
        norm_e = hilbert_schmidt_norm(Ah - A)
        norm_r = hilbert_schmidt_norm(A)

    return norm_e / norm_r


def no_parameters(model):
    '''Utility to retrieve number of trainable parameters in a model.

    Ref: Vadim Smolyakov
    [https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7]

    Parameters
    ----------
    model : nn.Module
        A PyTorch module.

    Returns
    -------
    Number of parameters in a model.
    '''
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def top_k_accuracy(output, target, k=5):
    '''Measures the accuracy of the top 'k' predictions.

    Parameters
    ----------
    output : torch.Tensor
        Output from model.

    target : torch.Tensor
        Actual values.

    Returns
    -------
    Top k accuracy metric.
    '''
    topk = output.topk(k, -1)[1].cpu()
    target_rep = torch.argmax(target, dim=-1)[...,None].repeat(1, k).cpu()
    _sum = torch.sum(torch.any(topk == target_rep, dim=-1))
    _len = len(output)
    return _sum / _len


def rolling_stats(data, min_max=False, winsize=101):
    '''Calculates rolling statistics on numpy array.

    Parameters
    ----------
    data : ArrayLike
        Current data.

    winsize : int (Optional)
        Window size.

    Returns
    -------
    Tuple of arrays, mean and standard deviation calculated over window.
    '''
    data = np.array(data)
    a = np.zeros_like(data)
    b = np.zeros_like(data)

    win = np.lib.stride_tricks.sliding_window_view(data, (winsize))

    if not min_max:
        a[winsize//2:-winsize//2+1] = np.mean(win, axis=-1)
        b[winsize//2:-winsize//2+1] = np.std(win, axis=-1)
        a[:winsize//2] = a[winsize//2]


    else:
        a[winsize//2:-winsize//2+1] = np.min(win, axis=-1)
        b[winsize//2:-winsize//2+1] = np.max(win, axis=-1)
        for arr in [a, b]:
            arr[:winsize//2] = arr[winsize//2]
            arr[-winsize//2:] = arr[-winsize//2]


    return a, b
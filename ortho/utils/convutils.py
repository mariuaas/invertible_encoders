from .. import (
    np, torch, fft, nn, math
)


def fftconvolve2d(x, h):
    '''Convolves two similar sized 2d signals by fft, corresponding to circular boundary
    conditions in nn.Conv2d.
    '''
    fx = fft.fft2(x)
    fh = fft.fft2(torch.flip(h, dims=(-2,-1)))
    m, n = fx.shape[-2:]
    fy = torch.einsum('bimn,oimn->bomn', fx, fh)
    y = fft.ifft2(fy).real
    return torch.roll(y, (-m//2+1, -n//2+1,), dims=(-2,-1))


def powkernel(h : torch.Tensor, n: int,  a: float = 1.0):
    '''Computes the n-th compositional convolution kernel.
    '''
    i, j = h.shape[-2:]

    p = (n - 1) * (i//2 + 1) % i
    q = (n - 1) * (j//2 + 1) % j

    if n == 0:
        h = torch.zeros_like(h)
        nn.init.dirac_(h)

    pr = (2,3,0,1)
    fh = fft.fft2(h).permute(*pr)
    fh = torch.linalg.matrix_power(a*fh, n).permute(*pr)
    h = fft.ifft2(fh).real

    return torch.roll(h, (p, q,), dims=(-2,-1))


def fackernel(h : torch.Tensor, n: int):
    '''Computes the n-th compositional convolution kernel.
    '''
    i, j = h.shape[-2:]

    p = (n - 1) * (i//2 + 1) % i
    q = (n - 1) * (j//2 + 1) % j

    if n == 0:
        h = torch.zeros_like(h)
        nn.init.dirac_(h)

    pr = (2,3,0,1)
    fh = fft.fft2(h).permute(*pr)
    fh = torch.linalg.matrix_power(fh, n).permute(*pr)
    fh /= math.factorial(n)
    h = fft.ifft2(fh).real

    return torch.roll(h, (p, q,), dims=(-2,-1))



def expkernel(h : torch.Tensor, n: int):
    '''Computes the n-th approximation to exponential convolution kernel.
    '''
    _h = torch.zeros_like(h)
    for k in range(n):
        _h += fackernel(h, k)
    return _h


def powserieskernel(h : torch.Tensor, n : int, a: float = 1.0):
    '''Computes the nth-order power series kernel for a convolution
    '''
    _h = torch.zeros_like(h)
    for k in range(n):
        _h += powkernel(h, k, a=a)
    return _h


def convert_coo_channel_values(k, v):
    '''Converts a kernel with value indices to full value list for COO matrix.
    '''
    return k[v[0], v[1], v[2], v[3]]


def _get_dims(ksize, xsize, pad, stride):
    '''Helper function to retrieve dimensions of loop.
    '''
    if isinstance(ksize, int):
        outsize = (xsize - ksize + 2*pad) // stride + 1
        loopsize = xsize - ksize + 2*pad + 1

    else:
        n = len(ksize)
        outsize = []
        loopsize = []
        for i in range(n):
            outsize.append((xsize[i] - ksize[i] + 2*pad[i]) // stride[i] + 1)
            loopsize.append(xsize[i] - ksize[i] + 2*pad[i] + 1)

    return outsize, loopsize

def _convert_param_dims(dim, ksize, xsize, pad, stride):
    '''Helper function to convert from int to iterables in conv2d.
    '''
    if isinstance(ksize, int):
        ksize = [ksize] * dim

    if isinstance(xsize, int):
        xsize = [xsize] * dim

    if isinstance(pad, int):
        pad = [pad] * dim

    if isinstance(stride, int):
        stride = [stride] * dim

    return ksize, xsize, pad, stride

def conv2d_coo(ksize, xsize, pad=1, stride=1):
    '''Generates coordinate and value indices for COO sparse 2D conv. matrix.
    '''
    ksize, xsize, pad, stride = _convert_param_dims(2, ksize, xsize, pad, stride)
    outsize, loopsize = _get_dims(ksize, xsize, pad, stride)
    out = []
    for R in range(loopsize[0]):
        for Ki in range(ksize[0]):
            for r in range(loopsize[1]):
                for ki in range(ksize[1]):
                    C = (Ki - pad[0] + R) % xsize[0]
                    c = (ki - pad[1] + r) % xsize[1]
                    if R % stride[0] == 0:
                        if r % stride[1] == 0:
                            rr = R // stride[0] * outsize[1] + r // stride[1]
                            cc = C * outsize[1] + c
                            out.append([Ki, ki, rr, cc])

    out = np.array(out)
    vals = out.T[:2]
    idxs = out.T[2:]
    full_outsize = np.prod(outsize)
    full_xsize = np.prod(xsize)
    return idxs, vals, (full_outsize, full_xsize)


def conv2d_coo_channels(in_feat, out_feat, ksize, xsize, pad=1, stride=1):
    '''Generates coordinate and value indices for COO sparse 2D conv. matrix w. channels.
    '''
    idx, val, shape = conv2d_coo(ksize, xsize, pad, stride)
    out_val = []
    out_idx = []

    for i in range(in_feat):
        for j in range(out_feat):
            cur_idx = np.vstack([
                idx[0] + i*shape[0],
                idx[1] + j*shape[1]
            ])
            cur_val = np.vstack([
                np.ones_like(val[0])*i,
                np.ones_like(val[0])*j,
                val
            ])
            out_val.append(cur_val)
            out_idx.append(cur_idx)

    out_val = np.hstack(out_val)
    out_idx = np.hstack(out_idx)
    full_shape = (out_feat * shape[0], in_feat * shape[1])
    return out_idx, out_val, full_shape


def conv_matrix(layer, xsize=None):
    #pylint: disable=access-member-before-definition
    if xsize is None:
        xsize = (
            layer.kernel_size[0] + layer.padding[0],
            layer.kernel_size[1] + layer.padding[1]
        )

    i, v, shape = conv2d_coo_channels(
        layer.in_channels, layer.out_channels, layer.kernel_size, xsize,
        pad=layer.padding, stride=layer.stride
    )

    v[0], v[1] = v[1].copy(), v[0].copy()
    i[0], i[1] = i[1].copy(), i[0].copy()

    kv = convert_coo_channel_values(layer.weight.data, v)
    return torch.sparse_coo_tensor(i, kv, shape, device=layer.weight.device)


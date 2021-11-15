from .. import (
    np, nn, torch, F, fft, modules, OrderedDict, List, Optional
)

from .dense import (
    _set_default_activation, _set_type_flags
)

def _construct_layer(dct, name, m, n, kernel, layer, diagonal, **kwargs):
    if diagonal:
        h = m + n + 2
        dct[name] = layer(m, h, kernel, **kwargs)
        dct[f'{name}_D'] = modules.Adjoint1x1Conv2d(h, n, **kwargs)
    else:
        dct[name] = layer(m, n, kernel, **kwargs)


def _construct_layer_dictionary(
        layer, final_layer, dims, kernels, diagonal, activation,
        activation_params, size, **kwargs
    ):
    dct = OrderedDict()
    dct['V'] = layer(dims[0], dims[1], kernels[0], **kwargs)
    dct['gamma0'] = activation(**activation_params)

    for i, (w, k) in enumerate(
        zip(zip(dims[1:-2], dims[2:-1]), kernels[1:-1])
    ):
        dct[f'W{i+1}'] = layer(w[0], w[1], k, **kwargs)
        dct[f'gamma{i+1}'] = activation(**activation_params)

    if diagonal:
        if diagonal == 'sep':
            dct['D'] = modules.AdjointSeperableDiagonal(
                size[0], size[1], **kwargs
            )
        elif diagonal == '1x1':
            dct['D'] = modules.Adjoint1x1Conv2d(
                dims[-2], dims[-2], size, **kwargs
            )
        elif diagonal == 'diag':
            dct['D'] = modules.AdjointConvDiagonal(
                dims[-2], size, **kwargs
            )

        else:
            raise ValueError(f'Got unexpected diagonal {diagonal}.')


    dct['U'] = final_layer(dims[-2], dims[-1], kernels[-1], **kwargs)

    return dct


class AdjointConv(nn.Module):

    def __init__(self, dims: List[int], kernels: List[int], size=None, bias=True, **kwargs) -> None:
        super().__init__()
        assert len(dims) > 2, f'No. dimensions must be d > 2, found {len(dims)}.'
        assert len(dims) == len(kernels) + 1, f'No. kernels must be one more than number of dims!'
        self.dims = dims
        self.kernels = kernels
        kwargs, self.activation, self.activation_params = _set_default_activation(**kwargs)
        kwargs, self.diagonal, self.stochastic = _set_type_flags(**kwargs)

        if 'type' in kwargs and kwargs['type'] == 'lie':
            layer = modules.UnitaryConv2d
        else:
            layer = modules.AdjointConv2d

        layer = layer if not bias else modules.AdjointAffineConv2d
        if self.stochastic:
            raise NotImplementedError('Nope!')

        else:
            final_layer = layer

        if self.diagonal:
            assert size is not None, 'Need size for diagonal layer!'
            assert self.diagonal in ['sep', 'diag', '1x1'], f'Invalid diagonal mode: {self.diagonal}.'

        dct = _construct_layer_dictionary(
            layer, final_layer, dims, kernels, self.diagonal, self.activation,
            self.activation_params, size, **kwargs
        )

        self.net = modules.AdjointSequential(dct)

        self.forward = self.net.forward
        self.T = self.net.T


class AEConv(nn.Module):

    def __init__(self, dims: List[int], kernels: List[int], size=None, bias=True, **kwargs) -> None:
        super().__init__()
        assert len(dims) > 2, f'No. dimensions must be d > 2, found {len(dims)}.'
        assert len(dims) == len(kernels) + 1, f'No. kernels must be one more than number of dims!'
        self.dims = dims
        self.kernels = kernels
        kwargs, self.activation, self.activation_params = _set_default_activation(**kwargs)
        kwargs, self.diagonal, self.stochastic = _set_type_flags(**kwargs)

        layer = modules.AdjointConv2d if not bias else modules.AdjointAffineConv2d
        if self.stochastic:
            raise NotImplementedError('Nope!')

        else:
            final_layer = layer

        if self.diagonal:
            assert size is not None, 'Need size for diagonal layer!'
            assert self.diagonal in ['sep', 'diag', '1x1'], f'Invalid diagonal mode: {self.diagonal}.'


        dct_fwd = _construct_layer_dictionary(
            layer, final_layer, dims, kernels, self.diagonal, self.activation,
            self.activation_params, size, **kwargs
        )

        self.net_fwd = modules.AdjointSequential(dct_fwd)


        dct_bck = _construct_layer_dictionary(
            layer, final_layer, dims, kernels, self.diagonal, self.activation,
            self.activation_params, size, **kwargs
        )

        self.net_bck = modules.AdjointSequential(dct_bck)


        self.forward = self.net_fwd.forward
        self.T = self.net_bck.T

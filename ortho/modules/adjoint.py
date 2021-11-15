from .. import (
    np, nn, torch, F, fft, OrderedDict, utils, Tuple, Union
)

# TODO: Add FFT and IFFT modules to be composed in sequential networks
# TODO: Finalize docstrings


def _set_init_params(a: float, b: float, **kwargs):
    '''Helper function to set initialization parameters for module classes.
    '''
    init_params = {'a': a}
    init = nn.init.kaiming_uniform_
    if b is not None:
        init = nn.init.uniform_
        init_params['b'] = b

    if 'init' in kwargs:
        init = kwargs['init']

        if 'init_params' not in kwargs:
            init_params = {}

        else:
            init_params = kwargs['init_params']

    return init, init_params



class AdjointSequential(nn.Sequential):
    """Adjoint Sequential Module

    Sequential module with added adjoint operator which applies adjoints of elements in
    reverse order.
    """

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def forward(self, input, *args, **kwargs):
        # NOTE: Appended optional arguments and keyword arguments to support cond.bias
        for module in self:
            input = module(input, *args, **kwargs)
        return input

    def T(self, input, *args, **kwargs):
        ''' The adjoint operator for the sequential block.
        '''
        for i, module in enumerate(reversed(self)):
            index = len(self) - i - 1
            assert \
                hasattr(module, "T") and callable(getattr(module, 'T')), \
                f'Module {module} at index {index} has no adjoint operator.'
            input = module.T(input, *args, **kwargs)
        return input


class AdjointPatchSplitter(nn.Module):
    """Adjoint patch splitting.

    TODO: docstring...
    """

    def __init__(self, split: Tuple[int, int], grid: Tuple[int, int]):
        super().__init__()
        self.split = split
        self.grid = grid


    def extra_repr(self) -> str:
        return 'split={}, grid={}'.format(
            self.split, self.grid
        )


    def forward(self, input, *args, **kwargs):
        return utils.winsplit(input, self.split)

    def T(self, input, *args, **kwargs):
        return utils.winunsplit(input, self.grid)


class AdjointPatchUnsplitter(nn.Module):
    """Reverse adjoint patch splitting.

    TODO: docstring...
    """

    def __init__(self, split: Tuple[int, int], grid: Tuple[int, int]):
        super().__init__()
        self._splitter = AdjointPatchSplitter(split, grid)
        self.split = split
        self.grid = grid

        self.forward = self._splitter.T
        self.T = self._splitter.forward


class AdjointFlatten(nn.Module):
    ''' Adjoint flatten module.
    '''

    def __init__(self, start_dim: int, end_dim: int, unflatten_size: Tuple[int, int]):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

        if start_dim > 0:
            self.dim = start_dim
        elif end_dim < 0:
            self.dim = end_dim
        else:
            raise ValueError(
                'Start dim is negative and end dim is negative! ' +
                'Cannot determine unflatten dimension...'
            )

        self._flat = nn.Flatten(start_dim, end_dim)
        self._unflat = nn.Unflatten(self.dim, unflatten_size)

    def forward(self, input, *args, **kwargs):
        return self._flat(input)

    def T(self, input, *args, **kwargs):
        return self._unflat(input)


class AdjointUnflatten(nn.Module):
    ''' Adjoint unflatten module.
    '''

    def __init__(self, start_dim: int, end_dim: int, unflatten_size: Tuple[int, int]):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

        if start_dim > 0:
            self.dim = start_dim
        elif end_dim < 0:
            self.dim = end_dim
        else:
            raise ValueError(
                'Start dim is negative and end dim is negative! ' +
                'Cannot determine unflatten dimension...'
            )

        self._flat = nn.Flatten(start_dim, end_dim)
        self._unflat = nn.Unflatten(self.dim, unflatten_size)

    def forward(self, input, *args, **kwargs):
        return self._unflat(input)

    def T(self, input, *args, **kwargs):
        return self._flat(input)


class AdjointPermutation(nn.Module):
    ''' Adjoint permutation module.
    '''

    def __init__(self, *permutation):
        super().__init__()
        self.permutation = permutation
        self._revperm = tuple(np.argsort(permutation))

    def forward(self, input, *args, **kwargs):
        return input.permute(*self.permutation)

    def T(self, input, *args, **kwargs):
        return input.permute(*self._revperm)


class Adjoint1x1Reshaper(nn.Module):
    '''Module to handle reshaping of 1x1 convolution
    '''

    def __init__(self, size, **kwargs):
        super().__init__()
        self.size = size
        self._flatten = AdjointFlatten(-2, -1, size)
        self._unflatten = AdjointUnflatten(-2, -1, size)
        self._permute = AdjointPermutation(0,2,1)

    def forward(self, input, *args, **kwargs):
        return self._permute(self._flatten(input)).unsqueeze(-1)

    def T(self, input, *args, **kwargs):
        return self._flatten.T(self._permute.T(input.squeeze(-1)))


class InvertableBatchNorm1d(nn.BatchNorm1d):
    """Batch Normalization with inverse method.

    Based on implementation by @ptrblck
    https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(InvertableBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def memoize_parameters(self, mean: torch.Tensor, var: torch.Tensor) -> None:
        self._curmean = mean
        self._curvar = var

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0])
            var = input.var([0], unbiased=False)
            self.memoize_parameters(mean, var)

            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean) / (torch.sqrt(var + self.eps))
        if self.affine:
            input = input * self.weight + self.bias

        return input


    def T(self, input):
        ''' The inverse operator for normalization.

        NOTE: This implementation relies on the forward operator having memoized the
        relevant parameters for the inverse operation during training.
        '''
        self._check_input_dim(input)

        if self.training:
            mean, var = self._curmean, self._curvar

        else:
            mean, var = self.running_mean, self.running_var

        if self.affine:
            input = (input - self.bias) / self.weight
        if self.affine:
            input = input * torch.sqrt(var + self.eps) + mean

        return input



class DIDownscale(nn.Module):
    '''Class to apply digit Interleaving to perform bijective channel contraction.
    '''

    def __init__(self, length, base=2, iterations=32, dtype=torch.double, sigmoid=True, squeeze=False):
        super().__init__()
        self.length = length
        self.base = base
        self.iter = iterations
        self.dtype = dtype
        self.sigmoid = sigmoid
        self.eps = torch.finfo(self.dtype).eps
        self.squeeze = squeeze

    def forward(self, x):
        if x.dtype != self.dtype and self.dtype == torch.double:
            x = x.double()

        if self.sigmoid:
            x = torch.sigmoid(x)

        y = utils.projection.do_digits(x, self.iter, self.base)

        if self.squeeze:
            return y.unsqueeze(-1).float()

        else:
            return y.float()

    def T(self, x):
        if x.dtype != self.dtype and self.dtype == torch.double:
            x = x.double()

        if self.squeeze:
            x = x.squeeze(-1)

        y = utils.projection.undo_digits(x, self.length, self.iter, self.base)

        if self.sigmoid:
            y = torch.special.logit(y.clip(self.eps, 1-self.eps))

        return y.float()


class DIUpscale(nn.Module):
    '''Class to apply digit Interleaving to perform bijective channel expansion.

    NOTE: Lower beta for increased stability.
    '''

    def __init__(self, length, base=2, iterations=32, dtype=torch.double, sigmoid=True, squeeze=False):
        super().__init__()
        self.length = length
        self.base = base
        self.iter = iterations
        self.dtype = dtype
        self.sigmoid = sigmoid
        self.squeeze = squeeze

    def forward(self, x):
        if x.dtype != self.dtype and self.dtype == torch.double:
            x = x.double()

        if self.sigmoid:
            x = torch.sigmoid(x)

        y =  utils.projection.undo_digits(x, self.length, self.iter, self.base)

        if self.squeeze:
            return y.unsqueeze(-1).float()

        else:
            return y.float()


    def T(self, x):
        if x.dtype != self.dtype and self.dtype == torch.double:
            x = x.double()

        if self.squeeze:
            x = x.squeeze(-1)

        y = utils.projection.do_digits(x, self.iter, self.base)

        if self.sigmoid:
            y = torch.special.logit(y)

        return y.float()


class GaussianTruncator(nn.Module):
    '''Class for conversion from truncated to non-truncated Gaussians.
    '''

    def __init__(self, mu=0, sd=1):
        self.stdnorm = torch.distributions.Normal(mu, sd)

    def cdf(self, z):
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z)
        return self.stdnorm.cdf(z)

    def icdf(self, z):
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z)
        return self.stdnorm.icdf(z)

    def ticdf(self, z, a, b):
        Pa = self.cdf(a)#[...,:z.shape[-1]])
        Pb = self.cdf(b)#[...,:z.shape[-1]])
        return self.icdf((z + Pa) * (Pb - Pa))

    def tcdf(self, z, a, b):
        Pa = self.cdf(a)#[...,:z.shape[-1]])
        Pb = self.cdf(b)#[...,:z.shape[-1]])
        return (self.cdf(z) - Pa) / (Pb - Pa)

    def truncate(self, z, a, b):
        return self.ticdf(self.cdf(z), a, b)

    def detruncate(self, z, a, b):
        return self.icdf(self.tcdf(z, a, b))



#####################################################################################
# The following classes and functions are included in this file for legacy purposes #
#####################################################################################


class AdjointMatrix(nn.Module):
    """Adjoint Matrix Layer.

    Dense Matrix Layer for Invertible Encoder-Decoders.
    """

    __constants__ = ['in_features', 'out_features']
    in_features : int
    out_features : int
    weight : torch.Tensor

    def __init__(self, in_features: int, out_features: int, **kwargs) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.init, self.init_params = _set_init_params(5**(1/2), None, **kwargs)
        factory_kwargs = utils.default_dictonary_getter(kwargs, device=None, dtype=None)
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.init(self.weight, **self.init_params)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )

    def forward(self, input, *args, **kwargs) -> torch.Tensor:
        '''The forward operator of the layer.
        '''
        return F.linear(input, self.weight, None)

    def T(self, input, *args, **kwargs) -> torch.Tensor:
        '''The adjoint operator of the layer.
        '''
        return F.linear(input, self.weight.T, None)


class ResolventOperator(nn.Module):
    """Simple Resolvent Operator for constructing Invertible Networks.
    """

    __constants__ = ['in_features', 'out_features']
    in_features : int
    out_features : int
    weight : torch.Tensor

    def __init__(self, in_features: int, out_features: int, a: float = 0.5, **kwargs) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.init, self.init_params = _set_init_params(5**(1/2), None, **kwargs)
        factory_kwargs = utils.default_dictonary_getter(kwargs, device=None, dtype=None)
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        self.a = a
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.init(self.weight, **self.init_params)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )

    def _liouville_neumann(self, n):
        weight = torch.zeros_like(self.weight)
        for k in range(n):
            weight += self.a**k * torch.linalg.matrix_power(self.weight, k)
        return weight

    def forward(self, input, *args, **kwargs) -> torch.Tensor:
        '''The forward operator of the layer.
        '''
        return input - self.a * F.linear(input, self.weight, None)

    def T(self, input, *args, n=12, **kwargs) -> torch.Tensor:
        '''The adjoint operator of the layer.
        '''
        return F.linear(input, self._liouville_neumann(n), None)


class AdjointUnitary(nn.Module):
    """Adjoint Unitary Matrix Layer.

    Maps parameterized skew symmetric matrix to Grassmann / Stiefel manifold. Gradient
    is then mapped from geodesic to parameters when computing gradients.
    """

    weight : torch.Tensor

    def __init__(self, features: int, type : str = 'lie', **kwargs):
        super().__init__()
        self.features = features
        factory_kwargs = utils.default_dictonary_getter(kwargs, device=None, dtype=None)
        self._weight = nn.Parameter(torch.zeros(features, features, **factory_kwargs))
        self.register_buffer(
            '_base',
            torch.zeros(features, features, **factory_kwargs)
        )
        self.initialize_base(**kwargs)
        assert type in ['lie', 'cayley'], f'Type {type} not supported.'
        self.type = type
        self.indices = utils.generate_skew_indices(features)

    @property
    def weight(self) -> torch.Tensor:
        M = self._weight.tril(-1)
        M = M - M.T
        if self.type == 'lie':
            return self._base @ torch.matrix_exp(M)

        elif self.type == 'cayley':
            I = torch.eye(self.features, device=self._weight.device)
            return self._base @ (I - M) @ torch.linalg.inv(I + M)

    @weight.setter
    def weight(self, value) -> torch.Tensor:
        with torch.no_grad():
            self._base.copy(value)
            self._weight = torch.zeros_like(value) # NOTE: This was wrong, how????

    def initialize_base(self, **kwargs):
        if 'base' in kwargs:
            self._base = kwargs['base']
        else:
            utils.init.haar_orthogonal_(self._base)

    def extra_repr(self) -> str:
        return 'features={}, type={}'.format(self.features, self.type)

    def forward(self, input, *args, **kwargs):
        return F.linear(input, self.weight, None)

    def T(self, input, *args, **kwargs):
        return F.linear(input, self.weight.T, None)



class AdjointSemiUnitary(nn.Module):
    """Adjoint Semi-Unitary Matrix Layer.

    Maps parameterized skew symmetric matrix to Grassmann / Stiefel manifold. Gradient
    is then mapped from tangent space to parameters when computing gradients.
    """

    weight : torch.Tensor

    def __init__(self, in_features: int, out_features: int, type: str = 'lie', **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = utils.default_dictonary_getter(kwargs, device=None, dtype=None)
        self._weight = nn.Parameter(
            torch.zeros(out_features, in_features, **factory_kwargs)
        )

        if out_features < in_features:
            self._underdetermined = True
            self.register_buffer(
                '_base',
                torch.zeros(in_features, in_features, **factory_kwargs)
            )

        else:
            self._underdetermined = False
            self.register_buffer(
                '_base',
                torch.zeros(out_features, out_features, **factory_kwargs)
            )

        self.initialize_base(**kwargs)
        self._nonsquare = in_features != out_features

        assert type in ['lie', 'cayley'], f'Type {type} not supported.'
        self.type = type

    @property
    def weight(self) -> torch.Tensor:
        M = self._weight.T if self._underdetermined else self._weight
        l, s = M.shape[0], M.shape[1]
        M = F.pad(M, (0, l - s)) if self._nonsquare else M
        M = M.tril(-1)
        M = M - M.T

        if self.type == 'lie':
            out = self._base @ torch.matrix_exp(M)

        elif self.type == 'cayley':
            I = torch.eye(l, device=self._weight.device)
            out =  self._base @ (I - M) @ torch.linalg.inv(I + M)

        else:
            raise ValueError('How did this happen?')

        return out[...,:s] if not self._underdetermined else out[...,:s].T

    @weight.setter
    def weight(self, value) -> torch.Tensor:
        l, s = self.out_features, self.in_features

        if self._underdetermined:
            value = value.T
            l, s = s, l

        with torch.no_grad():
            if self._nonsquare:
                Z = value.new_empty(l, l-s)
                Z.normal_()
                for _ in range(2):
                    Z = torch.linalg.qr(Z - value @ (value.T @ Z)).Q

                value = torch.cat([value, Z], dim=-1)

            self._base.copy(value)
            self._weight = torch.zeros_like(value)[...,:s]

    def initialize_base(self, **kwargs):
        if 'base' in kwargs:
            self._base = kwargs['base']
        else:
            utils.init.haar_orthogonal_(self._base)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, type={}'.format(
            self.in_features, self.out_features, self.type
        )

    def forward(self, input, *args, **kwargs):
        return F.linear(input, self.weight, None)

    def T(self, input, *args, **kwargs):
        return F.linear(input, self.weight.T, None)


class AdjointBias(nn.Module):

    """Adjoint Bias Layer.

    Bias Layer for Dense Invertible Encoder-Decoders.
    """

    __constants__ = ['features']
    features : int
    weight : torch.Tensor

    def __init__(self, features: int, **kwargs) -> None:
        super().__init__()
        self.features = features
        if 'init' in kwargs and kwargs['init'] in [nn.init.orthogonal_, nn.init.eye_]:
            del kwargs['init']
        self.init, self.init_params = _set_init_params(-1e-1, 1e-1, **kwargs)
        factory_kwargs = utils.default_dictonary_getter(kwargs, device=None, dtype=None)
        self.weight = nn.Parameter(torch.empty(features, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.init(self.weight, **self.init_params)

    def extra_repr(self) -> str:
        return f'features={self.features}'

    def forward(self, input, *args, **kwargs) -> torch.Tensor:
        '''The forward (addition) for the bias.
        '''
        return input + self.weight

    def T(self, input, *args, **kwargs) -> torch.Tensor:
        '''The adjoint (subtraction) of the bias.
        '''
        return input - self.weight


class AdjointDiagonal(nn.Module):

    """Adjoint Diagonal Layer.

    Adjoint Diagonal Layer for Dense Invertible Encoder-Decoders.
    """

    __constants__ = ['features']
    features : int
    weight : torch.Tensor

    def __init__(self, features: int, **kwargs) -> None:
        super().__init__()
        self.features = features
        if 'init' in kwargs and kwargs['init'] in [nn.init.orthogonal_, nn.init.eye_]:
            del kwargs['init']
        self.init, self.init_params = _set_init_params(.9, 1.1, **kwargs)
        factory_kwargs = utils.default_dictonary_getter(kwargs, device=None, dtype=None)
        self.weight = nn.Parameter(torch.empty(features, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.init(self.weight, **self.init_params)

    def extra_repr(self) -> str:
        return 'features={}'.format(
            self.features,
        )

    def diag(self) -> torch.Tensor:
        return torch.diag(self.weight)

    def invdiag(self) -> torch.Tensor:
        return torch.diag(1 / self.weight)

    def forward(self, input, *args, **kwargs) -> torch.Tensor:
        '''The forward diagonal operator.
        '''
        return input * self.weight

    def T(self, input, *args, **kwargs) -> torch.Tensor:
        '''The adjoint diagonal operator.
        '''
        return input / self.weight


class ConditionalAdjointBias(nn.Module):

    """Conditional Bias Layer

    Class for calculating conditional bias using a target vector.

    Attributes
    ----------
    classes : int
        The number of classes the layer can represent.
    features : int
        The number of features.
    """
    __constants__ = ['classes', 'features']
    features : int
    weight : torch.Tensor

    def __init__(self, features: int, classes: int, f=nn.Softmax(dim=-1), **kwargs) -> None:
        super().__init__()
        self.classes = classes
        self.features = features
        self.f = f if f is not None else lambda x: x
        self.init, self.init_params = _set_init_params(-1e-1, 1e-1, **kwargs)
        factory_kwargs = utils.default_dictonary_getter(kwargs, device=None, dtype=None)
        self.weight = nn.Parameter(torch.empty(classes, features, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.init(self.weight, **self.init_params)

    def extra_repr(self) -> str:
        return f'classes={self.classes}, features={self.features}'

    def forward(self, input, classes, *args, **kwargs) -> torch.Tensor:
        '''The forward (addition) for the bias.
        '''
        return input + self.f(classes) @ self.weight

    def T(self, input, classes, *args, **kwargs) -> torch.Tensor:
        '''The adjoint (subtraction) of the bias.
        '''
        output = input - self.f(classes) @ self.weight
        return output


class AdjointDenseAffine(nn.Module):

    """Adjoint Affine Dense Layer.

    A dense adjoint affine layer composed of a weight matrix and a bias term.
    """

    __constants__ = ['in_features', 'out_features']
    in_features : int
    out_features : int

    def __init__(self, in_features, out_features, **kwargs) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if 'resolvent' in  kwargs and kwargs['resolvent']:
            matrix = ResolventOperator(in_features, out_features)

        elif 'type' in kwargs and kwargs['type'] is not None:
            matrix = AdjointSemiUnitary(in_features, out_features, **kwargs)

        else:
            matrix = AdjointMatrix(in_features, out_features, **kwargs)

        self.affine = AdjointSequential(
            OrderedDict({
                'W': matrix,
                'b': AdjointBias(out_features, **kwargs)
            })
        )
        self.forward = self.affine.forward
        self.T = self.affine.T

    @property
    def weight(self):
        return self.affine.W.weight


class AdjointStochastic(nn.Module):
    '''Adjoint Stochastic (Gaussian) layer.

    Can in essence be any location scale family, but currently only supports
    Gaussian reparametrization.
    '''

    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.M = AdjointDenseAffine(in_features, out_features, **kwargs)
        self.S = AdjointDenseAffine(in_features, out_features, **kwargs)

    @property
    def weight(self):
        return self.M.W.weight

    def forward(self, input, *args, sample=True, **kwargs):
        mu = self.M(input)
        lv = self.S(input)

        if sample:
            return utils.reparametrize_gaussian(mu, lv), mu, lv

        else:
            return mu, mu, lv

    def T(self, input, *args, **kwargs):
        return self.M.T(input)


class ConditionalAdjointDenseAffine(nn.Module):

    """Adjoint Affine/Linear Layer.

    A dense adjoint affine layer composed of a weight matrix and a bias term.
    """

    __constants__ = ['in_features', 'out_features', 'classes']
    in_features : int
    out_features : int
    classes : int

    def __init__(self, in_features, out_features, classes, **kwargs) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.classes = classes
        self.affine = AdjointSequential(
            OrderedDict({
                'W': AdjointMatrix(in_features, out_features, **kwargs),
                'b': ConditionalAdjointBias(out_features, classes, **kwargs)
            })
        )
        self.forward = self.affine.forward
        self.T = self.affine.T

    @property
    def weight(self):
        return self.affine.W.weight


class ConditionalAdjointStochastic(nn.Module):

    def __init__(self, in_features, out_features, classes, **kwargs):
        super().__init__()
        self.M = ConditionalAdjointDenseAffine(in_features, out_features, classes, **kwargs)
        self.S = ConditionalAdjointDenseAffine(in_features, out_features, classes, **kwargs)

    @property
    def weight(self):
        return self.M.W.weight

    def forward(self, input, classes, *args, sample=True, **kwargs):
        mu = self.M(input, classes)
        lv = self.S(input, classes)

        if sample:
            return utils.reparametrize_gaussian(mu, lv), mu, lv

        else:
            return mu, mu, lv

    def T(self, input, classes, *args, **kwargs):
        return self.M.T(input, classes)







def _get_formula(input):
    d = input.dim()

    if d == 4:
        formula = 'pq,bcqr,rs->bcps'
    elif d == 3:
        formula = 'pq,bqr,rs->bps'
    elif d == 2:
        formula = 'pq,qr,rs->ps'
    else:
        raise ValueError(f'Dimension {d} not supported!')

    return formula

def _get_formula_bias2d(input, channels: int = 0):
    d = input.dim()

    if channels < 0:
        raise ValueError(f'Must have channels >= 0, got {channels}!')

    if d == 2:
        # Add if for weight dimensions
        if channels > 0:
            formula = 'bp,pcmn->bcmn'
        elif channels == 0:
            formula = 'bp,pmn->bmn'
    elif d == 1:
        if channels > 0:
            formula = 'p,pcmn->cmn'
        elif channels == 0:
            formula = 'p,pmn->mn'
    else:
        raise ValueError(f'Dimension {d} not supported!')

    return formula


class AdjointMixer(nn.Module):
    '''Base mixer layer for adjoint network composition.
    '''

    __constants__ = [
        'in_features_m',
        'in_features_n',
        'out_features_m',
        'out_features_n',
    ]

    in_features_m : int
    in_features_n : int
    out_features_m : int
    out_features_n : int

    weight_U : torch.Tensor
    weight_V : torch.Tensor

    def __init__(
        self,
        in_features_m: int,
        out_features_m: int,
        in_features_n: int,
        out_features_n: int,
        **kwargs
    ) -> None:
        super().__init__()
        self.in_features_m = in_features_m
        self.out_features_m = out_features_m
        self.in_features_n = in_features_n
        self.out_features_n = out_features_n
        self.init, self.init_params = _set_init_params(5**(1/2), None, **kwargs)
        factory_kwargs = utils.default_dictonary_getter(kwargs, device=None, dtype=None)
        self.weight_U = nn.Parameter(torch.empty(out_features_m, in_features_m, **factory_kwargs))
        self.weight_V = nn.Parameter(torch.empty(in_features_n, out_features_n, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        self.init(self.weight_U, **self.init_params)
        self.init(self.weight_V, **self.init_params)

    def extra_repr(self) -> str:
        return 'in_features_m={}, in_features_n={}, out_features_m={}, out_features_n={}'.format(
            self.in_features_m, self.in_features_n, self.out_features_m, self.out_features_n
        )

    def forward(self, input, *args, **kwargs) -> torch.Tensor:
        '''The forward operator of the layer.
        '''
        formula = _get_formula(input)
        return torch.einsum(formula, self.weight_U, input, self.weight_V)

    def T(self, input, *args, **kwargs) -> torch.Tensor:
        '''The adjoint operator of the layer.
        '''
        formula = _get_formula(input)
        return torch.einsum(formula, self.weight_U.T, input, self.weight_V.T)


class AdjointUnitaryMixer(nn.Module):
    '''Base mixer layer for adjoint network composition.
    '''

    __constants__ = [
        'in_features_m',
        'in_features_n',
        'out_features_m',
        'out_features_n',
    ]

    in_features_m : int
    in_features_n : int
    out_features_m : int
    out_features_n : int

    weight_U : torch.Tensor
    weight_V : torch.Tensor

    def __init__(
        self,
        in_features_m: int,
        out_features_m: int,
        in_features_n: int,
        out_features_n: int,
        type='lie',
        **kwargs
    ) -> None:
        super().__init__()
        self.in_features_m = in_features_m
        self.out_features_m = out_features_m
        self.in_features_n = in_features_n
        self.out_features_n = out_features_n
        factory_kwargs = utils.default_dictonary_getter(kwargs, device=None, dtype=None)

        self.weight_U = AdjointSemiUnitary(in_features_m, out_features_m, type=type, **factory_kwargs)
        self.weight_V = AdjointSemiUnitary(out_features_n, in_features_n, type=type, **factory_kwargs)

    def extra_repr(self) -> str:
        return 'in_features_m={}, in_features_n={}, out_features_m={}, out_features_n={}'.format(
            self.in_features_m, self.in_features_n, self.out_features_m, self.out_features_n
        )

    def forward(self, input, *args, **kwargs) -> torch.Tensor:
        '''The forward operator of the layer.
        '''
        formula = _get_formula(input)
        return torch.einsum(formula, self.weight_U.weight, input, self.weight_V.weight)

    def T(self, input, *args, **kwargs) -> torch.Tensor:
        '''The adjoint operator of the layer.
        '''
        formula = _get_formula(input)
        return torch.einsum(formula, self.weight_U.weight.T, input, self.weight_V.weight.T)


class ResolventMixer(nn.Module):
    '''Base mixer layer for adjoint network composition.
    '''

    __constants__ = [
        'in_features_m',
        'in_features_n',
        'out_features_m',
        'out_features_n',
    ]

    in_features_m : int
    in_features_n : int
    out_features_m : int
    out_features_n : int

    weight_U : torch.Tensor
    weight_V : torch.Tensor

    def __init__(
        self,
        in_features_m: int,
        out_features_m: int,
        in_features_n: int,
        out_features_n: int,
        a: float = 0.5,
        **kwargs
    ) -> None:
        super().__init__()
        self.in_features_m = in_features_m
        self.out_features_m = out_features_m
        self.in_features_n = in_features_n
        self.out_features_n = out_features_n
        assert in_features_m == out_features_m, "Resolvent needs full rank!"
        assert in_features_n == out_features_n, "Resolvent needs full rank!"
        self.init, self.init_params = _set_init_params(5**(1/2), None, **kwargs)
        factory_kwargs = utils.default_dictonary_getter(kwargs, device=None, dtype=None)
        self.weight_U = nn.Parameter(torch.empty(out_features_m, in_features_m, **factory_kwargs))
        self.weight_V = nn.Parameter(torch.empty(in_features_n, out_features_n, **factory_kwargs))
        self.a = a
        self.reset_parameters()

    def reset_parameters(self):
        self.init(self.weight_U, **self.init_params)
        self.init(self.weight_V, **self.init_params)

    def extra_repr(self) -> str:
        return 'in_features_m={}, in_features_n={}, out_features_m={}, out_features_n={}'.format(
            self.in_features_m, self.in_features_n, self.out_features_m, self.out_features_n
        )

    def _liouville_neumann(self, n):
        weight_U = torch.zeros_like(self.weight_U)
        weight_V = torch.zeros_like(self.weight_V)
        for k in range(n):
            weight_U += self.a**k * torch.linalg.matrix_power(self.weight_U, k)
            weight_V += self.a**k * torch.linalg.matrix_power(self.weight_V, k)
        return weight_U, weight_V

    def forward(self, input, *args, **kwargs) -> torch.Tensor:
        '''The forward operator of the layer.
        '''
        formula = _get_formula(input)
        return input - self.a * torch.einsum(formula, self.weight_U, input, self.weight_V)

    def T(self, input, *args, n=12, **kwargs) -> torch.Tensor:
        '''The adjoint operator of the layer.
        '''
        formula = _get_formula(input)
        weight_U, weight_V = self._liouville_neumann(n)
        return torch.einsum(formula, weight_U, input, weight_V)


class AdjointBias2d(nn.Module):

    """Adjoint Bias Layer for 2d data.

    Bias Layer for Dense Invertible Encoder-Decoders.
    """

    __constants__ = ['features_m', 'features_n']
    features_m : int
    features_n : int
    weight : torch.Tensor

    def __init__(self, features_m: int, features_n: int, **kwargs) -> None:
        super().__init__()
        self.features_m = features_m
        self.features_n = features_n
        self.init, self.init_params = _set_init_params(-1e-1, 1e-1, **kwargs)
        factory_kwargs = utils.default_dictonary_getter(kwargs, device=None, dtype=None)
        self.weight = nn.Parameter(torch.empty(features_m, features_n, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.init(self.weight, **self.init_params)

    def extra_repr(self) -> str:
        return f'features_m={self.features_m}, features_n={self.features_n}'

    def forward(self, input, *args, **kwargs) -> torch.Tensor:
        '''The forward (addition) for the bias.
        '''
        return input + self.weight

    def T(self, input, *args, **kwargs) -> torch.Tensor:
        '''The adjoint (subtraction) of the bias.
        '''
        return input - self.weight


class AdjointSeperableDiagonal(nn.Module):
    """Adjoint Seperable Diagonal Layer.

    Adjoint Seperable Diagonal Layer for Dense Invertible Encoder-Decoders.
    """

    __constants__ = ['features']
    features : int
    weight : torch.Tensor

    def __init__(self, features_m: int, features_n: int, **kwargs) -> None:
        super().__init__()
        self.features_m = features_m
        self.features_n = features_n
        if 'init' in kwargs and kwargs['init'] in [nn.init.orthogonal_, nn.init.eye_]:
            del kwargs['init']
        self.init, self.init_params = _set_init_params(.9, 1.1, **kwargs)
        factory_kwargs = utils.default_dictonary_getter(kwargs, device=None, dtype=None)
        self.weight_m = nn.Parameter(torch.empty(features_m, **factory_kwargs))
        self.weight_n = nn.Parameter(torch.empty(features_n, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.init(self.weight_m, **self.init_params)
        self.init(self.weight_n, **self.init_params)

    @staticmethod
    def _get_formula(input):
        d = input.dim()

        if d == 4:
            formula = 'q,bcqr,r->bcqr'
        elif d == 3:
            formula = 'q,bqr,r->bqr'
        elif d == 2:
            formula = 'q,qr,r->qr'
        else:
            raise ValueError(f'Dimension {d} not supported!')

        return formula

    def extra_repr(self) -> str:
        return 'features_m={}, features_n={}'.format(
            self.features_m, self.features_n,
        )

    def forward(self, input, *args, **kwargs) -> torch.Tensor:
        '''The forward diagonal operator.
        '''
        formula = self._get_formula(input)
        return torch.einsum(formula, self.weight_m, input, self.weight_n)

    def T(self, input, *args, **kwargs) -> torch.Tensor:
        '''The adjoint diagonal operator.
        '''
        formula = self._get_formula(input)
        return torch.einsum(formula, 1/self.weight_m, input, 1/self.weight_n)


class ConditionalAdjointBias2d(nn.Module):

    """Conditional Bias Layer

    Class for calculating conditional bias using a target vector.

    TODO: Add support for channels.

    Attributes
    ----------
    classes : int
        The number of classes the layer can represent.
    features_m : int
        The number of patch features.
    features_n : int
        The dimension of each patch.
    """
    __constants__ = ['classes', 'features_m', 'features_n']
    features_m : int
    features_n : int
    classes : int
    weight : torch.Tensor

    def __init__(
        self,
        features_m: int,
        features_n: int,
        classes : int,
        f = nn.Softmax(dim=-1),
        **kwargs
    ) -> None:
        super().__init__()
        self.classes = classes
        self.features_m = features_m
        self.features_n = features_n
        self.f = f if f is not None else lambda x: x

        self.init, self.init_params = _set_init_params(-1e-1, 1e-1, **kwargs)
        factory_kwargs = utils.default_dictonary_getter(kwargs, device=None, dtype=None)
        self.weight = nn.Parameter(torch.empty(classes, features_m, features_n, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.init(self.weight, **self.init_params)

    def extra_repr(self) -> str:
        return f'features_m={self.features_m}, features_n={self.features_n}, classes={self.classes}'

    def forward(self, input, classes, *args, **kwargs) -> torch.Tensor:
        '''The forward (addition) for the bias.
        '''
        formula = _get_formula_bias2d(classes)
        return input + torch.einsum(formula, self.f(classes), self.weight).view(*input.shape)

    def T(self, input, classes, *args, **kwargs) -> torch.Tensor:
        '''The adjoint (subtraction) of the bias.
        '''
        formula = _get_formula_bias2d(classes)
        return input - torch.einsum(formula, self.f(classes), self.weight).view(*input.shape)


class AdjointAffineMixer(nn.Module):

    """Adjoint Affine Mixer Layer.

    A dense adjoint affine layer composed of a weight matrix and a bias term.
    """

    __constants__ = [
        'in_features_m',
        'in_features_n',
        'out_features_m'
        'out_features_n'
    ]
    in_features_m : int
    out_features_m : int
    in_features_n : int
    out_features_n : int

    def __init__(
        self,
        in_features_m: int,
        out_features_m: int,
        in_features_n: int,
        out_features_n: int,
        **kwargs
    ) -> None:
        super().__init__()
        self.in_features_m = in_features_m
        self.out_features_m = out_features_m
        self.in_features_n = in_features_n
        self.out_features_n = out_features_n

        if 'resolvent' in kwargs and kwargs['resolvent']:
            mixer = ResolventMixer(
                in_features_m,
                out_features_m,
                in_features_n,
                out_features_n,
                **kwargs
            )


        elif 'type' in kwargs and kwargs['type'] is not None:
            mixer = AdjointUnitaryMixer(
                in_features_m,
                out_features_m,
                in_features_n,
                out_features_n,
                **kwargs
            )

        else:
            mixer = AdjointMixer(
                in_features_m,
                out_features_m,
                in_features_n,
                out_features_n,
                **kwargs
            )

        self.affine = AdjointSequential(
            OrderedDict({
                'W': mixer,
                'b': AdjointBias2d(
                    out_features_m,
                    out_features_n,
                    **kwargs
                )
            })
        )
        self.forward = self.affine.forward
        self.T = self.affine.T

    @property
    def weight(self):
        return self.affine.W.weight_U, self.affine.W.weight_V


class ConditionalAdjointAffineMixer(nn.Module):

    """Conditional Adjoint Affine Mixer Layer.
    """

    __constants__ = [
        'classes',
        'in_features_m',
        'in_features_n',
        'out_features_m'
        'out_features_n'
    ]
    classes : int
    in_features_m : int
    out_features_m : int
    in_features_n : int
    out_features_n : int

    def __init__(
        self,
        in_features_m: int,
        out_features_m: int,
        in_features_n: int,
        out_features_n: int,
        classes : int,
        **kwargs
    ) -> None:
        super().__init__()
        self.classes = classes
        self.in_features_m = in_features_m
        self.out_features_m = out_features_m
        self.in_features_n = in_features_n
        self.out_features_n = out_features_n
        self.affine = AdjointSequential(
            OrderedDict({
                'W': AdjointMixer(
                        in_features_m,
                        out_features_m,
                        in_features_n,
                        out_features_n,
                        **kwargs
                ),
                'b': ConditionalAdjointBias2d(
                    out_features_m,
                    out_features_n,
                    classes,
                    **kwargs
                )
            })
        )
        self.forward = self.affine.forward
        self.T = self.affine.T

    @property
    def weight(self):
        return self.affine.W.weight_U, self.affine.W.weight_V


class AdjointStochasticMixer(nn.Module):

    def __init__(
        self,
        in_features_m: int,
        out_features_m: int,
        in_features_n: int,
        out_features_n: int,
        **kwargs
    ):
        super().__init__()
        self.M = AdjointAffineMixer(
            in_features_m,
            out_features_m,
            in_features_n,
            out_features_n,
            **kwargs
        )
        self.S = AdjointAffineMixer(
            in_features_m,
            out_features_m,
            in_features_n,
            out_features_n,
            **kwargs
        )

    @property
    def weight(self):
        return self.M.W.weight

    def forward(self, input, *args, sample=True, **kwargs):
        mu = self.M(input)
        lv = self.S(input)

        if sample:
            return utils.reparametrize_gaussian(mu, lv), mu, lv

        else:
            return mu, mu, lv

    def T(self, input, *args, **kwargs):
        return self.M.T(input)


class ConditionalAdjointStochasticMixer(nn.Module):

    def __init__(
        self,
        in_features_m: int,
        out_features_m: int,
        in_features_n: int,
        out_features_n: int,
        classes : int,
        **kwargs
    ):
        super().__init__()
        self.M = ConditionalAdjointAffineMixer(
            in_features_m,
            out_features_m,
            in_features_n,
            out_features_n,
            classes,
            **kwargs
        )
        self.S = ConditionalAdjointAffineMixer(
            in_features_m,
            out_features_m,
            in_features_n,
            out_features_n,
            classes,
            **kwargs
        )

    @property
    def weight(self):
        return self.M.W.weight

    def forward(self, input, classes, *args, sample=True, **kwargs):
        mu = self.M(input, classes)
        lv = self.S(input, classes)

        if sample:
            return utils.reparametrize_gaussian(mu, lv), mu, lv

        else:
            return mu, mu, lv

    def T(self, input, classes, *args, **kwargs):
        return self.M.T(input, classes)


class AdjointConcurrentMixer(nn.Module):

    def __init__(
        self,
        in_features_m_a,
        out_features_m_a,
        in_features_n_a,
        out_features_n_a,
        in_features_m_b,
        out_features_m_b,
        in_features_n_b,
        out_features_n_b,
        **kwargs
    ):
        super().__init__()
        self.A = AdjointAffineMixer(
            in_features_m_a,
            out_features_m_a,
            in_features_n_a,
            out_features_n_a,
            **kwargs
        )
        self.B = AdjointMixer(
            in_features_m_b,
            out_features_m_b,
            in_features_n_b,
            out_features_n_b,
            **kwargs
        )

        self.beta = AdjointAffineMixer(
            out_features_m_a,
            out_features_m_b,
            out_features_n_a,
            out_features_n_b,
            **kwargs
        )

    def forward(self, input_a, input_b, *args, **kwargs):
        a = self.A(input_a)
        b = self.B(input_b) + self.beta(a)
        return a, b

    def T(self, input_a, input_b, *args, **kwargs):
        a = self.A.T(input_a)
        b = self.B.T(input_b - self.beta(input_a))
        return a, b


class AdjointStochasticConcurrentMixer(nn.Module):

    def __init__(
        self,
        in_features_m_a,
        out_features_m_a,
        in_features_n_a,
        out_features_n_a,
        in_features_m_b,
        out_features_m_b,
        in_features_n_b,
        out_features_n_b,
        **kwargs
    ):
        super().__init__()

        self.A = AdjointAffineMixer(
            in_features_m_a,
            out_features_m_a,
            in_features_n_a,
            out_features_n_a,
            **kwargs
        )
        self.M = AdjointMixer(
            in_features_m_b,
            out_features_m_b,
            in_features_n_b,
            out_features_n_b,
            **kwargs
        )
        self.S = AdjointMixer(
            in_features_m_b,
            out_features_m_b,
            in_features_n_b,
            out_features_n_b,
            **kwargs
        )
        self.beta_m = AdjointAffineMixer(
            out_features_m_a,
            out_features_m_b,
            out_features_n_a,
            out_features_n_b,
            **kwargs
        )
        self.beta_s = AdjointAffineMixer(
            out_features_m_a,
            out_features_m_b,
            out_features_n_a,
            out_features_n_b,
            **kwargs
        )

    @property
    def weight(self):
        raise NotImplementedError('sorrz, not ready bruh')

    def forward(self, input_a, input_b, *args, sample=True, **kwargs):
        a = self.A(input_a)
        mu = self.M(input_b) + self.beta_m(a)
        lv = self.S(input_b) + self.beta_s(a)

        if sample:
            return a, utils.reparametrize_gaussian(mu, lv), mu, lv

        else:
            return a, mu, mu, lv

    def T(self, input_a, input_b, *args, **kwargs):
        a = self.A.T(input_a)
        b = self.M.T(input_b - self.beta_m(input_a))
        return a, b


class AdjointConcurrentDiagonal(nn.Module):

    def __init__(
        self,
        features_m_a,
        features_n_a,
        features_m_b,
        features_n_b,
        **kwargs
    ):
        super().__init__()
        self.DA = AdjointSeperableDiagonal(
            features_m_a, features_n_a, **kwargs
        )
        self.DB = AdjointSeperableDiagonal(
            features_m_b, features_n_b, **kwargs
        )

    def forward(self, input_a, input_b, *args, **kwargs):
        a = self.DA(input_a)
        b = self.DB(input_b)
        return a, b

    def T(self, input_a, input_b, *args, **kwargs):
        a = self.DA.T(input_a)
        b = self.DB.T(input_b)
        return a, b

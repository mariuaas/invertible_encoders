from .. import (
    np, nn, torch, F, OrderedDict, utils, Tuple, Union
)

from .adjoint import(
    _set_init_params, AdjointSequential
)

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

    def __init__(self, in_features: int, out_features: int, lmbda: float = 1.0, **kwargs) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.init, self.init_params = _set_init_params(5**(1/2), None, **kwargs)
        factory_kwargs = utils.default_dictonary_getter(kwargs, device=None, dtype=None)
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        self.lmbda = lmbda
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
            weight += self.lmbda**k * torch.linalg.matrix_power(self.weight, k)
        return weight

    def forward(self, input, *args, **kwargs) -> torch.Tensor:
        '''The forward operator of the layer.
        '''
        return input - self.lmbda * F.linear(input, self.weight, None)

    def T(self, input, *args, n=12, **kwargs) -> torch.Tensor:
        '''The adjoint operator of the layer.
        '''
        return F.linear(input, self._liouville_neumann(n), None)


class AdjointUnitary(nn.Module):
    """Adjoint Unitary Matrix Layer.

    Maps parameterized skew symmetric matrix to SO(n). Gradient
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
            self._weight = torch.zeros_like(value)

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
            matrix = ResolventOperator(in_features, out_features, **kwargs)

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
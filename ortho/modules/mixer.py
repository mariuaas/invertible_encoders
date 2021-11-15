from .. import (
    np, nn, torch, F, fft, OrderedDict, utils, Tuple, Union
)

from .adjoint import (
    _set_init_params, AdjointSequential,
)

from .dense import (
    AdjointSemiUnitary
)

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
        lmbda: float = 1.0,
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
        self.lmbda = lmbda
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
            weight_U += self.lmbda**k * torch.linalg.matrix_power(self.weight_U, k)
            weight_V += self.lmbda**k * torch.linalg.matrix_power(self.weight_V, k)
        return weight_U, weight_V

    def forward(self, input, *args, **kwargs) -> torch.Tensor:
        '''The forward operator of the layer.
        '''
        input = torch.transpose(input - self.lmbda * self.weight_U @ input, -2, -1)
        input = torch.transpose(input - self.lmbda * self.weight_V @ input, -2, -1)
        return input

    def T(self, input, *args, n=12, **kwargs) -> torch.Tensor:
        '''The adjoint operator of the layer.
        '''
        weight_U, weight_V = self._liouville_neumann(n)
        input = torch.transpose(weight_U @ input, -2, -1)
        input = torch.transpose(weight_V @ input, -2, -1)
        return input


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

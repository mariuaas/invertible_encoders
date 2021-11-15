from .. import (
    np, nn, torch, F, fft, OrderedDict, utils, Tuple, Union
)

from .adjoint import (
    AdjointFlatten, AdjointUnflatten, _set_init_params, AdjointSequential,
    AdjointPermutation, Adjoint1x1Reshaper
)

from .dense import (
    AdjointBias,
    AdjointDiagonal
)

def _check_conv_kwargs(**kwargs):
    from types import SimpleNamespace
    if isinstance(kwargs['kernel_size'], int):
        padding = kwargs['kernel_size'] // 2

    elif isinstance(kwargs['kernel_size'], tuple):
        padding = (
            kwargs['kernel_size'][0] // 2,
            kwargs['kernel_size'][1] // 2
        )
    else:
        ValueError(f'Kernel size must be int or tuple, got {kwargs["kernel_size"]}.')

    c = SimpleNamespace(
        **utils.default_dictonary_getter(
            kwargs,
            in_channels=None,
            out_channels=None,
            kernel_size=None,
            stride=1,
            padding=padding,
            transposed=False,
            dilation=1,
            output_padding=None,
            groups=1,
            padding_mode='circular',
            resolvent=False,
            lmbda=1.0,
            iter=12,
        )
    )
    if c.in_channels % c.groups != 0:
        raise ValueError('in_channels must be divisible by groups')

    if c.out_channels % c.groups != 0:
        raise ValueError('out_channels must be divisible by groups')

    valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
    if c.padding_mode not in valid_padding_modes:
        raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
            valid_padding_modes, c.padding_mode))

    if c.padding_mode != 'zeros' and c.transposed:
        raise ValueError(f'Transposed convolution requires padding mode zeros, got {c.padding_mode}')

    return c.__dict__


def _check_unitary_conv_kwargs(**kwargs):
    from types import SimpleNamespace
    if isinstance(kwargs['kernel_size'], int):
        padding = kwargs['kernel_size'] // 2

    elif isinstance(kwargs['kernel_size'], tuple):
        padding = (
            kwargs['kernel_size'][0] // 2,
            kwargs['kernel_size'][1] // 2
        )
    else:
        raise ValueError(f'Kernel size must be int or tuple, got {kwargs["kernel_size"]}.')

    if 'padding_mode' in kwargs and kwargs['padding_mode'] != 'circular':
        raise ValueError(f'Only circular padding implemented!')

    c = SimpleNamespace(
        **utils.default_dictonary_getter(
            kwargs,
            in_channels=None,
            out_channels=None,
            kernel_size=None,
            stride=1,
            padding=padding,
            dilation=1,
            output_padding=None,
            groups=1,
            padding_mode='circular',
            iter=16,
        )
    )
    if c.in_channels % c.groups != 0:
        raise ValueError('in_channels must be divisible by groups')

    if c.out_channels % c.groups != 0:
        raise ValueError('out_channels must be divisible by groups')

    return c.__dict__


class AdjointConv2d(nn.Module):

    __constants__ = [
        'stride', 'padding', 'dilation', 'groups',
        'padding_mode', 'in_channels',
        'out_channels', 'kernel_size'
    ]

    in_channels : int
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Union[str, Tuple[int, ...]]
    dilation: Tuple[int, ...]
    groups: int
    padding_mode: str
    weight: torch.Tensor

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.init, self.init_params = _set_init_params(5**.5, None, **kwargs)
        factory_kwargs = utils.default_dictonary_getter(kwargs, device=None, dtype=None)
        self.__dict__.update(
            _check_conv_kwargs(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = kernel_size,
                **kwargs
            )
        )
        self.kernel_size = tuple([self.kernel_size] * 2) if isinstance(self.kernel_size, int) else self.kernel_size
        self.stride = tuple([self.stride] * 2) if isinstance(self.stride, int) else self.stride
        self.padding = tuple([self.padding] * 2) if isinstance(self.padding, int) else self.padding
        self.dilation = tuple([self.dilation] * 2) if isinstance(self.dilation, int) else self.dilation
        self._reversed_padding_repeated_twice = tuple(x for x in reversed(self.padding) for _ in range(2))

        self.weight = nn.Parameter(
            torch.empty(
                (out_channels, in_channels // self.groups, *self.kernel_size),
                **factory_kwargs
            )
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.init(self.weight, **self.init_params)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def pad(self, input):
        if self.padding_mode == 'zeros':
            return input, self.padding
        else:
            return F.pad(
                input,
                self._reversed_padding_repeated_twice,
                mode = self.padding_mode
            ), (0, 0)

    def forward(self, input, *args, **kwargs):
        padinput, padding = self.pad(input)

        if self.resolvent:
            return input - self.lmbda * F.conv2d(
                padinput,
                self.weight,
                None,
                self.stride,
                padding,
                self.dilation,
                self.groups
            )

        return F.conv2d(
            padinput,
            self.weight,
            None,
            self.stride,
            padding,
            self.dilation,
            self.groups
        )

    def T(self, input, *args, **kwargs):
        if self.resolvent:
            if 'iter' in kwargs:
                iter = kwargs['iter']
            else:
                iter = self.iter

            weight = utils.convutils.powserieskernel(self.weight, iter, self.lmbda)
            input, padding = self.pad(input)
            return F.conv2d(
                input,
                weight,
                None,
                self.stride,
                padding,
                self.dilation,
                self.groups
            )

        if not self.transposed:
            input, padding = self.pad(input)
            return F.conv2d(
                input,
                torch.flip(self.weight, (2,3)).permute([1,0,2,3]),
                None,
                self.stride,
                padding,
                self.dilation,
                self.groups
            )
        else:
            return F.conv_transpose2d(
                input,
                self.weight,
                None,
                self.stride,
                self.padding,
                (self.stride[0] - 1, self.stride[1] - 1),
                self.groups,
                self.dilation,
            )


class UnitaryConv2d(nn.Module):

    __constants__ = [
        'stride', 'padding', 'dilation', 'groups',
        'padding_mode', 'in_channels',
        'out_channels', 'kernel_size'
    ]

    in_channels : int
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Union[str, Tuple[int, ...]]
    dilation: Tuple[int, ...]
    groups: int
    padding_mode: str
    weight: torch.Tensor

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.init, self.init_params = _set_init_params(5**.5, None, **kwargs)
        factory_kwargs = utils.default_dictonary_getter(kwargs, device=None, dtype=None)
        self.__dict__.update(
            _check_unitary_conv_kwargs(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = kernel_size,
                **kwargs
            )
        )
        self.kernel_size = tuple([self.kernel_size] * 2) if isinstance(self.kernel_size, int) else self.kernel_size
        self.stride = tuple([self.stride] * 2) if isinstance(self.stride, int) else self.stride
        self.padding = tuple([self.padding] * 2) if isinstance(self.padding, int) else self.padding
        self.dilation = tuple([self.dilation] * 2) if isinstance(self.dilation, int) else self.dilation
        self._reversed_padding_repeated_twice = tuple(x for x in reversed(self.padding) for _ in range(2))

        self.register_buffer(
            '_base',
            torch.zeros(
                (out_channels, in_channels // self.groups, *self.kernel_size),
                **factory_kwargs
            )
        )

        self._weight = nn.Parameter(
            torch.zeros(
                (out_channels, in_channels // self.groups, *self.kernel_size),
                **factory_kwargs
            )
        )
        self.initialize_base()

    @property
    def weight(self) -> torch.Tensor:
        M = self._weight - self._weight.flip(-2, -1)
        E = utils.convutils.expkernel(M, self.iter)
        return utils.convutils.fftconvolve2d(self._base, E)

    @weight.setter
    def weight(self, value) -> torch.Tensor:
        with torch.no_grad():
            self._base.copy(value)
            self._weight = torch.zeros_like(value)

    def initialize_base(self):
        nn.init.orthogonal_(self._base)
        skew = self._base - self._base.flip(-2, -1)
        self._base = utils.convutils.expkernel(skew, self.iter)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def pad(self, input):
        return F.pad(
            input,
            self._reversed_padding_repeated_twice,
            mode = self.padding_mode
        ), (0, 0)

    def forward(self, input, *args, **kwargs):
        padinput, padding = self.pad(input)
        return F.conv2d(
            padinput,
            self.weight,
            None,
            self.stride,
            padding,
            self.dilation,
            self.groups
        )

    def T(self, input, *args, **kwargs):
        input, padding = self.pad(input)
        return F.conv2d(
            input,
            torch.flip(self.weight, (-2,-1)).permute([1,0,2,3]),
            None,
            self.stride,
            padding,
            self.dilation,
            self.groups
        )


class Adjoint1x1Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, size, **kwargs):
        super().__init__()
        self.in_chennels = in_channels
        self.out_channels = out_channels
        factory_kwargs = utils.default_dictonary_getter(kwargs, device=None, dtype=None)
        self._reshaper = Adjoint1x1Reshaper(size)
        self._k = in_channels - out_channels + 1
        self._n = size[0]*size[1]
        self.conv = AdjointConv2d(
            self._n,
            self._n,
            (self._k, 1),
            stride=1,
            padding=(0,0),
            init=nn.init.uniform_,
            init_params={'a':0.9/self._n , 'b':1.1/self._n},
            **factory_kwargs
        )

    @property
    def weight(self):
        return self.conv.weight

    def forward(self, input, *args, **kwargs):
        return self._reshaper.T(
            self.conv(
                self._reshaper(input)
            )
        )

    def T(self, input, *args, **kwargs):
        return self._reshaper.T(
            self.conv.T(
                self._reshaper(input)
            )
        )


class AdjointConvDiagonal(nn.Module):

    def __init__(self, channels, size, **kwargs):
        super().__init__()
        self.channels = channels
        self.size = size
        factory_kwargs = utils.default_dictonary_getter(kwargs, device=None, dtype=None)
        self._flatten = AdjointFlatten(-3, -1, (channels, *size))
        self.diag = AdjointDiagonal(size[0] * size[1] * channels, **factory_kwargs)

    def forward(self, input, *args, **kwargs):
        return self._flatten.T(self.diag(self._flatten(input)))

    def T(self, input, *args, **kwargs):
        return self._flatten.T(self.diag.T(self._flatten(input)))


class AdjointAffineConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = tuple([kernel_size]*2) if isinstance(kernel_size, int) else kernel_size
        if 'type' in kwargs and kwargs['type'] == 'lie':
            layer = UnitaryConv2d
        else:
            layer = AdjointConv2d

        self.affine = AdjointSequential(
            OrderedDict(
                W = layer(in_channels, out_channels, kernel_size, **kwargs),
                b = AdjointSequential(
                    AdjointPermutation(0,2,3,1),
                    AdjointBias(out_channels, **kwargs),
                    AdjointPermutation(0,3,1,2)
                )
            )
        )
        self.forward = self.affine.forward
        self.T = self.affine.T

    @property
    def weight(self):
        return self.affine.W.weight


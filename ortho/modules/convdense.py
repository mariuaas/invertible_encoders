from .. import (
    nn, torch, OrderedDict
)

from .adjoint import (
    AdjointSequential, AdjointPatchSplitter, AdjointPatchUnsplitter,
    AdjointFlatten, AdjointUnflatten, AdjointPermutation
)

from .mixer import (
    AdjointAffineMixer
)

from .conv import (
    AdjointAffineConv2d
)

from .activation import (
    BiCELU
)

class AdjointPatchConv(nn.Module):
    '''Seperable Patch Layer with Resolvent Convolutions
    '''

    def __init__(
        self,
        dims_in,
        dims_out,
        channels_in,
        channels_out,
        split,
        kernel_size=3,
        type=None,
        activation=BiCELU,
        order='B',
        **kwargs
    ):
        super().__init__()
        self.dims_in = dims_in
        self.dims_out = dims_out
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.split = split

        self._n_in = dims_in // split
        self._m_in = dims_in // self._n_in

        self._n_out = dims_out // split
        self._m_out = dims_out // self._n_out

        self.order = order

        dct = OrderedDict()

        dct['splitter'] = AdjointSequential(
            AdjointPatchSplitter(
                (self._n_in,)*2,
                (self._m_in,)*2,
            ),
            AdjointFlatten(-2, -1, (self._n_in,)*2),
            AdjointFlatten(-3, -2, (channels_in, self._m_in**2)),
        )

        dct['mixer'] = AdjointAffineMixer(
            channels_in * self._m_in**2,
            channels_out * self._m_out**2,
            self._n_in**2,
            self._n_out**2,
            type=type,
            **kwargs
        )

        dct['unsplitter'] = AdjointSequential(
            AdjointUnflatten(-3, -2, (channels_out, self._m_out**2)),
            AdjointUnflatten(-2, -1, (self._n_out,)*2),
            AdjointPatchUnsplitter(
                (self._n_out,)*2,
                (self._m_out,)*2
            ),
        )

        self.dense = AdjointSequential(dct)


        self.conv = AdjointAffineConv2d(
            channels_in if order=='A' else channels_out,
            channels_in if order=='A' else channels_out,
            kernel_size,
            padding=kernel_size//2,
            padding_mode='circular',
            type='lie',
            iter=12,
            **kwargs
        )

        self.activation = activation()


    def forward(self, input):

        if self.order == 'A':
            input = self.conv(input)
            input = self.activation(input)
            input = self.dense(input)

        else:
            input = self.dense(input)
            input = self.activation(input)
            input = self.conv(input)

        return input


    def T(self, input):

        if self.order == 'A':
            input = self.dense.T(input)
            input = self.activation.T(input)
            input = self.conv.T(input)

        else:
            input = self.conv.T(input)
            input = self.activation.T(input)
            input = self.dense.T(input)

        return input


class AdjointPatchConv2(nn.Module):
    '''Seperable Patch Layer with Resolvent Convolutions.

    In this layer, the channel expansion is performed by expanding and contracting
    the patches instead of the channels. This provided better results in other
    experiments.
    '''

    def __init__(
        self,
        dims_in,
        dims_out,
        channels_in,
        channels_out,
        split,
        kernel_size=3,
        type=None,
        activation=BiCELU,
        order='B',
        **kwargs
    ):
        super().__init__()
        self.dims_in = dims_in
        self.dims_out = dims_out
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.split = split

        self._n_in = dims_in // split
        self._m_in = dims_in // self._n_in

        self._n_out = dims_out // split
        self._m_out = dims_out // self._n_out

        self.order = order

        dct = OrderedDict()

        dct['splitter'] = AdjointSequential(
            AdjointPatchSplitter(
                (self._n_in,)*2,
                (self._m_in,)*2,
            ),
            AdjointFlatten(-2, -1, (self._n_in,)*2),
            AdjointPermutation(0,1,3,2),
            AdjointFlatten(-3, -2, (channels_in, self._n_in**2)),
            AdjointPermutation(0,2,1),
        )

        dct['mixer'] = AdjointAffineMixer(
            self._m_in**2,
            self._m_out**2,
            channels_in * self._n_in**2,
            channels_out * self._n_out**2,
            type=type,
            **kwargs
        )

        dct['unsplitter'] = AdjointSequential(
            AdjointPermutation(0,2,1),
            AdjointUnflatten(-3, -2, (channels_out, self._n_out**2)),
            AdjointPermutation(0,1,3,2),
            AdjointUnflatten(-2, -1, (self._n_out,)*2),
            AdjointPatchUnsplitter(
                (self._n_out,)*2,
                (self._m_out,)*2
            ),
        )

        self.dense = AdjointSequential(dct)

        self.conv = AdjointAffineConv2d(
            channels_in if order=='A' else channels_out,
            channels_in if order=='A' else channels_out,
            kernel_size,
            padding=kernel_size//2,
            padding_mode='circular',
            resolvent=True,
            **kwargs
        )

        self.activation = activation()


    def forward(self, input):

        if self.order == 'A':
            input = self.conv(input)
            input = self.activation(input)
            input = self.dense(input)

        else:
            input = self.dense(input)
            input = self.activation(input)
            input = self.conv(input)

        return input


    def T(self, input):

        if self.order == 'A':
            input = self.dense.T(input)
            input = self.activation.T(input)
            input = self.conv.T(input)

        else:
            input = self.conv.T(input)
            input = self.activation.T(input)
            input = self.dense.T(input)

        return input



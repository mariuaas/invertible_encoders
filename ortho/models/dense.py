from .. import (
    np, nn, torch, F, fft, modules, OrderedDict, List
)

def _set_default_activation(**kwargs):
    if 'activation' in kwargs:
        activation = kwargs['activation']
        del kwargs['activation']

    else:
        activation = modules.BiCELU

    if 'activation_params' in kwargs:
        activation_params = kwargs['activation_params']
        del kwargs['activation_params']

    else:
        activation_params = {}

    return kwargs, activation, activation_params


def _set_type_flags(**kwargs):
    if 'diagonal' in kwargs:
        diagonal = kwargs['diagonal']
        del kwargs['diagonal']

    else:
        diagonal = True

    if 'stochastic' in kwargs:
        stochastic = bool(kwargs['stochastic'])
        del kwargs['stochastic']

    else:
        stochastic = False

    return kwargs, diagonal, stochastic


def _construct_layer_dictionary(
        layer, final_layer, dims, diagonal, activation, activation_params, **kwargs
    ):
    dct = OrderedDict()

    if 'flatten' in kwargs:
        dct['flat'] = modules.AdjointFlatten(*kwargs['flatten'])
    dct['V'] = layer(dims[0], dims[1], **kwargs)
    dct['gamma0'] = activation(**activation_params)

    for i, w in enumerate(zip(dims[1:-2], dims[2:-1])):
        dct[f'W{i+1}'] = layer(w[0], w[1], **kwargs)
        dct[f'gamma{i+1}'] = activation(**activation_params)

    if diagonal:
        dct['D'] = modules.AdjointDiagonal(dims[-2], **kwargs)

    dct['U'] = final_layer(dims[-2], dims[-1], **kwargs)

    if 'flatten' in kwargs:
        dct['unflat'] = modules.AdjointUnflatten(*kwargs['flatten'])

    return dct


def _construct_layer_dictionary_mixer(
        layer, final_layer, patch_dims, grid_dims, diagonal, activation,
        activation_params, **kwargs
    ):
    dct = OrderedDict()

    if 'splitter' in kwargs:
        split = kwargs['splitter'][:2]
        flat = kwargs['splitter'][2:]
        dct['split'] = modules.AdjointSequential(
            OrderedDict({
                'split': modules.AdjointPatchSplitter(*split),
                'flat': modules.AdjointFlatten(*flat)
            })
        )

    dct['V'] = layer(
        patch_dims[0],
        patch_dims[1],
        grid_dims[0],
        grid_dims[1],
        **kwargs
    )
    dct['gamma0'] = activation(**activation_params)

    for i, ((pd1, pd2), (gd1, gd2)) in enumerate(zip(
        zip(patch_dims[1:-2], patch_dims[2:-1]),
        zip(grid_dims[1:-2], grid_dims[2:-1])
    )):
        dct[f'W{i+1}'] = layer(pd1, pd2, gd1, gd2, **kwargs)
        dct[f'gamma{i+1}'] = activation(**activation_params)

    if diagonal:
        dct['D'] = modules.AdjointSeperableDiagonal(
            patch_dims[-2],
            grid_dims[-2],
            **kwargs
        )

    dct['U'] = final_layer(
        patch_dims[-2],
        patch_dims[-1],
        grid_dims[-2],
        grid_dims[-1],
        **kwargs
    )

    if 'unsplitter' in kwargs:
        split = kwargs['unsplitter'][:2]
        flat = kwargs['unsplitter'][2:]
        dct['unsplit'] = modules.AdjointSequential(
            OrderedDict({
                'flat': modules.AdjointUnflatten(*flat),
                'split': modules.AdjointPatchUnsplitter(*split)
            })
        )

    return dct

def _construct_concurrent_layer_dictionary(
        layer, final_layer, patch_dims_A, grid_dims_A, patch_dims_B, grid_dims_B,
        diagonal, activation, activation_params, **kwargs
):
    dct = OrderedDict()
    dct['V'] = layer(
        patch_dims_A[0],
        patch_dims_A[1],
        grid_dims_A[0],
        grid_dims_A[1],
        patch_dims_B[0],
        patch_dims_B[1],
        grid_dims_B[0],
        grid_dims_B[1],
        **kwargs
    )
    dct['gamma0'] = modules.ConcurrentActivationWrapper(
        activation(**activation_params)
    )

    for i, (((pda1, pda2), (gda1, gda2)), ((pdb1, pdb2), (gdb1, gdb2))) in enumerate(zip(
        zip(
            zip(patch_dims_A[1:-2], patch_dims_A[2:-1]),
            zip(grid_dims_A[1:-2], grid_dims_A[2:-1])
        ),
        zip(
            zip(patch_dims_B[1:-2], patch_dims_B[2:-1]),
            zip(grid_dims_B[1:-2], grid_dims_B[2:-1])
        ),
    )):
        dct[f'W{i+1}'] = layer(
            pda1, pda2, gda1, gda2, pdb1, pdb2, gdb1, gdb2, **kwargs
        )
        dct[f'gamma{i+1}'] = modules.ConcurrentActivationWrapper(
            activation(**activation_params)
        )

    if diagonal:
        dct['D'] = modules.AdjointConcurrentDiagonal(
            patch_dims_A[-2],
            grid_dims_A[-2],
            patch_dims_B[-2],
            grid_dims_B[-2],
            **kwargs
        )

    dct['U'] = final_layer(
        patch_dims_A[-2],
        patch_dims_A[-1],
        grid_dims_A[-2],
        grid_dims_A[-1],
        patch_dims_B[-2],
        patch_dims_B[-1],
        grid_dims_B[-2],
        grid_dims_B[-1],
        **kwargs
    )

    return dct



class AdjointDense(nn.Module):

    def __init__(self, dims: List[int], **kwargs) -> None:
        super().__init__()
        assert len(dims) > 2, f'No. dimensions must be d > 2, found {len(dims)}.'
        self.dims = dims
        kwargs, self.activation, self.activation_params = _set_default_activation(**kwargs)
        kwargs, self.diagonal, self.stochastic = _set_type_flags(**kwargs)

        # TODO: Possibly clear used args after extraction

        layer = modules.AdjointDenseAffine
        if self.stochastic:
            final_layer = modules.AdjointStochastic

        else:
            final_layer = layer

        dct = _construct_layer_dictionary(
            layer, final_layer, dims, self.diagonal, self.activation,
            self.activation_params, **kwargs
        )

        self.net = modules.AdjointSequential(dct)

        self.forward = self.net.forward
        self.T = self.net.T

    def get_weight_dict(self):
        """Extracts a dictionary with relevant weight matrices.
        """
        dct = OrderedDict()
        dct['V'] = self.net.V.weight
        for i, _ in enumerate(self.dims[1:-2]):
            layer = self.net.__getattr__(f'W{i+1}')
            dct[f'W{i+1}'] = layer.weight

        if self.diagonal:
            dct['D'] = self.net.D.diag()

        if self.stochastic:
            dct['U'] = self.net.U.M.weight

        else:
            dct['U'] = self.net.U.weight

        dct_bck = OrderedDict()
        for key, value in reversed(dct.items()):
            if key != 'D':
                dct_bck[f'{key}.T'] = value.T
            else:
                dct_bck[f'{key}.T'] = torch.diag(1/torch.diag(value.T))

        return OrderedDict({**dct, **dct_bck})


class AEDense(nn.Module):

    def __init__(self, dims: List[int], **kwargs) -> None:
        super(AEDense, self).__init__()
        assert len(dims) > 2, f'No. dimensions must be d > 2, found {len(dims)}.'
        self.dims = dims
        kwargs, self.activation, self.activation_params = _set_default_activation(**kwargs)
        kwargs, self.diagonal, self.stochastic = _set_type_flags(**kwargs)
        layer = modules.AdjointDenseAffine
        if self.stochastic:
            final_layer = modules.AdjointStochastic

        else:
            final_layer = layer

        dct_fwd = _construct_layer_dictionary(
            layer, final_layer, dims, self.diagonal, self.activation,
            self.activation_params, **kwargs
        )

        dct_bck = _construct_layer_dictionary(
            layer, layer, dims, self.diagonal, self.activation,
            self.activation_params, **kwargs
        )

        self.net_fwd = modules.AdjointSequential(dct_fwd)
        self.net_bck = modules.AdjointSequential(dct_bck)
        self.forward = self.net_fwd.forward
        self.T = self.net_bck.T


    def get_weight_dict(self):
        """Extracts a dictionary with relevant weight matrices.
        """
        dct = OrderedDict()
        dct['V'] = self.net_fwd.V.weight
        for i, _ in enumerate(self.dims[1:-2]):
            layer = self.net_fwd.__getattr__(f'W{i+1}')
            dct[f'W{i+1}'] = layer.weight

        if self.diagonal:
            dct['D'] = self.net_fwd.D.diag()

        if self.stochastic:
            dct['U'] = self.net_fwd.U.M.weight

        else:
            dct['U'] = self.net_fwd.U.weight

        dct['U.T'] = self.net_bck.U.weight.T

        if self.diagonal:
            dct['D.T'] = self.net_bck.D.invdiag()

        for i in reversed(range(len(self.dims[1:-2]))):
            layer = self.net_bck.__getattr__(f'W{i+1}')
            dct[f'W{i+1}.T'] = layer.weight.T

        dct['V.T'] = self.net_bck.V.weight.T

        return dct



class AdjointDenseMixer(nn.Module):

    def __init__(self, patch_dims: List[int], grid_dims: List[int], **kwargs):
        super().__init__()
        assert len(patch_dims) > 2, f'No. dimensions must be d > 2, found {len(patch_dims)}.'
        assert len(grid_dims) == len(patch_dims), f'Dimensions must match, found {len(grid_dims)} and {len(patch_dims)}.'

        self.patch_dims = patch_dims
        self.grid_dims = grid_dims
        kwargs, self.activation, self.activation_params = _set_default_activation(**kwargs)
        kwargs, self.diagonal, self.stochastic = _set_type_flags(**kwargs)

        layer = modules.AdjointAffineMixer
        if self.stochastic:
            final_layer = modules.AdjointStochasticMixer
        else:
            final_layer = layer

        dct = _construct_layer_dictionary_mixer(
            layer, final_layer, patch_dims, grid_dims, self.diagonal, self.activation,
            self.activation_params, **kwargs
        )

        self.net = modules.AdjointSequential(dct)

        self.forward = self.net.forward
        self.T = self.net.T

    def get_weight_dict(self):
        """Extracts a dictionary with relevant weight matrices.
        """
        raise NotImplementedError('Not implemented for Mixer...')


class AEDenseMixer(nn.Module):

    def __init__(self, patch_dims: List[int], grid_dims: List[int], **kwargs):
        super().__init__()
        assert len(patch_dims) > 2, f'No. dimensions must be d > 2, found {len(patch_dims)}.'
        assert len(grid_dims) == len(patch_dims), f'Dimensions must match, found {len(grid_dims)} and {len(patch_dims)}.'

        self.patch_dims = patch_dims
        self.grid_dims = grid_dims
        kwargs, self.activation, self.activation_params = _set_default_activation(**kwargs)
        kwargs, self.diagonal, self.stochastic = _set_type_flags(**kwargs)

        layer = modules.AdjointAffineMixer
        if self.stochastic:
            raise NotImplementedError('Variational not implemented')

        else:
            final_layer = layer

        dct_fwd = _construct_layer_dictionary_mixer(
            layer, final_layer, patch_dims, grid_dims, self.diagonal, self.activation,
            self.activation_params, **kwargs
        )

        dct_bck = _construct_layer_dictionary_mixer(
            layer, layer, patch_dims, grid_dims, self.diagonal, self.activation,
            self.activation_params, **kwargs
        )

        self.net_fwd = modules.AdjointSequential(dct_fwd)
        self.net_bck = modules.AdjointSequential(dct_bck)
        self.forward = self.net_fwd.forward
        self.T = self.net_bck.T


    def get_weight_dict(self):
        """Extracts a dictionary with relevant weight matrices.
        """
        raise NotImplementedError('Not implemented for Mixer...')


class ConditionalAdjointDense(nn.Module):

    def __init__(self, dims: List[int], classes, **kwargs) -> None:
        super().__init__()
        assert len(dims) > 2, f'No. dimensions must be d > 2, found {len(dims)}.'
        self.dims = dims
        self.classes = classes
        kwargs, self.activation, self.activation_params = _set_default_activation(**kwargs)
        kwargs, self.diagonal, self.stochastic = _set_type_flags(**kwargs)

        # TODO: Possibly clear used args after extraction

        layer = modules.ConditionalAdjointDenseAffine
        if self.stochastic:
            final_layer = modules.ConditionalAdjointStochastic

        else:
            final_layer = layer

        kwargs['classes'] = classes

        dct = _construct_layer_dictionary(
            layer, final_layer, dims, self.diagonal, self.activation,
            self.activation_params, **kwargs
        )

        self.net = modules.AdjointSequential(dct)

        self.forward = self.net.forward
        self.T = self.net.T

    def get_weight_dict(self):
        """Extracts a dictionary with relevant weight matrices.
        """
        dct = OrderedDict()
        dct['V'] = self.net.V.weight
        for i, _ in enumerate(self.dims[1:-2]):
            layer = self.net.__getattr__(f'W{i+1}')
            dct[f'W{i+1}'] = layer.weight

        if self.diagonal:
            dct['D'] = self.net.D.diag()

        if self.stochastic:
            dct['U'] = self.net.U.M.weight

        else:
            dct['U'] = self.net.U.weight

        dct_bck = OrderedDict()
        for key, value in reversed(dct.items()):
            if key != 'D':
                dct_bck[f'{key}.T'] = value.T
            else:
                dct_bck[f'{key}.T'] = torch.diag(1/torch.diag(value.T))

        return OrderedDict({**dct, **dct_bck})



class ConditionalAEDense(nn.Module):

    def __init__(self, dims: List[int], classes, **kwargs) -> None:
        super().__init__()
        assert len(dims) > 2, f'No. dimensions must be d > 2, found {len(dims)}.'
        self.dims = dims
        self.classes = classes
        kwargs, self.activation, self.activation_params = _set_default_activation(**kwargs)
        kwargs, self.diagonal, self.stochastic = _set_type_flags(**kwargs)

        # TODO: Possibly clear used args after extraction

        layer = modules.ConditionalAdjointDenseAffine
        if self.stochastic:
            final_layer = modules.ConditionalAdjointStochastic

        else:
            final_layer = layer

        kwargs['classes'] = classes

        dct_fwd = _construct_layer_dictionary(
            layer, final_layer, dims, self.diagonal, self.activation,
            self.activation_params, **kwargs
        )

        dct_bck = _construct_layer_dictionary(
            layer, layer, dims, self.diagonal, self.activation,
            self.activation_params, **kwargs
        )

        self.net_fwd = modules.AdjointSequential(dct_fwd)
        self.net_bck = modules.AdjointSequential(dct_bck)
        self.forward = self.net_fwd.forward
        self.T = self.net_bck.T

    def get_weight_dict(self):
        """Extracts a dictionary with relevant weight matrices.
        """
        dct = OrderedDict()
        dct['V'] = self.net_fwd.V.weight
        for i, _ in enumerate(self.dims[1:-2]):
            layer = self.net_fwd.__getattr__(f'W{i+1}')
            dct[f'W{i+1}'] = layer.weight

        if self.diagonal:
            dct['D'] = self.net_fwd.D.diag()

        if self.stochastic:
            dct['U'] = self.net_fwd.U.M.weight

        else:
            dct['U'] = self.net_fwd.U.weight

        dct['U.T'] = self.net_bck.U.weight.T

        if self.diagonal:
            dct['D.T'] = self.net_bck.D.invdiag()

        for i in reversed(range(len(self.dims[1:-2]))):
            layer = self.net_bck.__getattr__(f'W{i+1}')
            dct[f'W{i+1}.T'] = layer.weight.T

        dct['V.T'] = self.net_bck.V.weight.T

        return dct


class AdjointConcurrentMixer(nn.Module):

    def __init__(
        self,
        patch_dims_A: List[int],
        grid_dims_A: List[int],
        patch_dims_B: List[int],
        grid_dims_B: List[int],
        **kwargs
    ):
        super().__init__()
        assert len(patch_dims_A) > 2, f'No. dimensions must be d > 2, found {len(patch_dims_A)}.'
        assert len(patch_dims_B) > 2, f'No. dimensions must be d > 2, found {len(patch_dims_B)}.'
        assert len(grid_dims_A) == len(patch_dims_A), f'Dimensions must match, found {len(grid_dims_A)} and {len(patch_dims_A)}.'
        assert len(grid_dims_B) == len(patch_dims_B), f'Dimensions must match, found {len(grid_dims_B)} and {len(patch_dims_B)}.'

        self.patch_dims_A = patch_dims_A
        self.patch_dims_B = patch_dims_B
        self.grid_dims_A = grid_dims_A
        self.grid_dims_B = grid_dims_B

        kwargs, self.activation, self.activation_params = _set_default_activation(**kwargs)
        kwargs, self.diagonal, self.stochastic = _set_type_flags(**kwargs)

        layer = modules.AdjointConcurrentMixer
        if self.stochastic:
            final_layer = modules.AdjointStochasticConcurrentMixer
        else:
            final_layer = layer

        dct = _construct_concurrent_layer_dictionary(
            layer, final_layer, patch_dims_A, grid_dims_A, patch_dims_B, grid_dims_B,
            self.diagonal, self.activation, self.activation_params, **kwargs
        )

        self.net = modules.AdjointSequential(dct)


    def forward(self, input_a, input_b, *args, **kwargs):
        for i, layer in enumerate(self.net):
            if not self.stochastic or i < len(self.net) - 1:
                input_a, input_b = layer(input_a, input_b, *args, **kwargs)

            else:
                input_a, input_b, mu, lv = layer(input_a, input_b, *args, **kwargs)

        if self.stochastic:
            return input_a, input_b, mu, lv

        else:
            return input_a, input_b

    def T(self, input_a, input_b, *args, **kwargs):
        for layer in reversed(self.net):
            input_a, input_b = layer.T(input_a, input_b, *args, **kwargs)
        return input_a, input_b


    def get_weight_dict(self):
        """Extracts a dictionary with relevant weight matrices.
        """
        raise NotImplementedError('Not implemented for Mixer...')


#####################################################################################


# class AdjointDenseVariational(nn.Module):
#     ''' NOTE: Deprecated. Use AdjointDense with Stochastic flag enabled.
#     '''

#     def __init__(self, dims: List[int], **kwargs) -> None:
#         super(AdjointDenseVariational, self).__init__()
#         assert len(dims) > 2, f'No. dimensions must be d > 2, found {len(dims)}.'
#         self.dims = dims
#         self.activation, self.activation_params = _set_default_activation(**kwargs)
#         layer = modules.AdjointDenseAffine
#         dct = OrderedDict()

#         # Construct layer for each dimension.
#         dct['V'] = layer(dims[0], dims[1], **kwargs)
#         dct['gamma0'] = self.activation(**self.activation_params)

#         for i, w in enumerate(zip(dims[1:-2], dims[2:-1])):
#             dct[f'W{i+1}'] = layer(w[0], w[1], **kwargs)
#             dct[f'gamma{i+1}'] = self.activation(**self.activation_params)

#         dct['U'] = modules.AdjointStochastic(dims[-2], dims[-1], **kwargs)

#         self.net = modules.AdjointSequential(dct)

#         self.forward = self.net.forward
#         self.T = self.net.T


#     def get_weight_dict(self):
#         """Extracts a dictionary with relevant weight matrices.
#         """
#         dct = OrderedDict()
#         dct['V'] = self.net.V.weight
#         for i, _ in enumerate(self.dims[1:-2]):
#             layer = self.net.__getattr__(f'W{i+1}')
#             dct[f'W{i+1}'] = layer.weight
#         dct['U'] = self.net.U.weight

#         dct_bck = OrderedDict()
#         for key, value in reversed(dct.items()):
#             dct_bck[f'{key}.T'] = value.T

#         return OrderedDict({**dct, **dct_bck})


# class AdjointDenseDiagonal(nn.Module):
#     ''' NOTE: Deprecated. Use AdjointDense with Diagonal flag enabled.
#     '''

#     def __init__(self, dims: List[int], **kwargs) -> None:
#         super(AdjointDenseDiagonal, self).__init__()
#         assert len(dims) > 2, f'No. dimensions must be d > 2, found {len(dims)}.'
#         self.dims = dims
#         self.activation, self.activation_params = _set_default_activation(**kwargs)
#         layer = modules.AdjointDenseAffine
#         dct = OrderedDict()

#         # Construct layer for each dimension.
#         dct['V'] = layer(dims[0], dims[1], **kwargs)
#         dct['gamma0'] = self.activation(**self.activation_params)

#         for i, w in enumerate(zip(dims[1:-2], dims[2:-1])):
#             dct[f'W{i+1}'] = layer(w[0], w[1], **kwargs)
#             dct[f'gamma{i+1}'] = self.activation(**self.activation_params)

#         dct['S'] = modules.AdjointDiagonal(dims[-2], **kwargs)
#         dct['U'] = layer(dims[-2], dims[-1], **kwargs)

#         self.net = modules.AdjointSequential(dct)

#         self.forward = self.net.forward
#         self.T = self.net.T

#     def get_weight_dict(self):
#         """Extracts a dictionary with relevant weight matrices.
#         """
#         dct = OrderedDict()
#         dct['V'] = self.net.V.weight
#         for i, _ in enumerate(self.dims[1:-2]):
#             layer = self.net.__getattr__(f'W{i+1}')
#             dct[f'W{i+1}'] = layer.weight
#         dct['U'] = self.net.U.weight

#         dct_bck = OrderedDict()
#         for key, value in reversed(dct.items()):
#             dct_bck[f'{key}.T'] = value.T

#         return OrderedDict({**dct, **dct_bck})


# class AEDense(nn.Module):

#     def __init__(self, dims: List[int], **kwargs) -> None:
#         super(AEDense, self).__init__()
#         assert len(dims) > 2, f'No. dimensions must be d > 2, found {len(dims)}.'
#         self.dims = dims
#         self.activation, self.activation_params = _set_default_activation(**kwargs)
#         layer = modules.AdjointDenseAffine
#         dct_fwd = OrderedDict()
#         dct_bck = OrderedDict()

#         # Construct layer for each dimension in forward model.
#         dct_fwd['V'] = layer(dims[0], dims[1], **kwargs)
#         dct_fwd['gamma0'] = self.activation(**self.activation_params)

#         for i, w in enumerate(zip(dims[1:-2], dims[2:-1])):
#             dct_fwd[f'W{i+1}'] = layer(w[0], w[1], **kwargs)
#             dct_fwd[f'gamma{i+1}'] = self.activation(**self.activation_params)

#         dct_fwd['U'] = layer(dims[-2], dims[-1], **kwargs)

#         # Construct layer for each dimension in adjoint/inverse model.
#         # NOTE: These are added in the opposite "forward" direction, and we use
#         #       the adjoint method T(input) to call the network in this direction.
#         dct_bck['V'] = layer(dims[0], dims[1], **kwargs)
#         dct_bck['gamma0'] = self.activation(**self.activation_params)

#         for i, w in enumerate(zip(dims[1:-2], dims[2:-1])):
#             dct_bck[f'W{i+1}'] = layer(w[0], w[1], **kwargs)
#             dct_bck[f'gamma{i+1}'] = self.activation(**self.activation_params)

#         dct_bck['U'] = layer(dims[-2], dims[-1], **kwargs)

#         self.net_fwd = modules.AdjointSequential(dct_fwd)
#         self.net_bck = modules.AdjointSequential(dct_bck)
#         self.forward = self.net_fwd.forward
#         self.T = self.net_bck.T


#     def get_weight_dict(self):
#         """Extracts a dictionary with relevant weight matrices.
#         """
#         dct = OrderedDict()
#         dct['V'] = self.net_fwd.V.weight
#         for i, _ in enumerate(self.dims[1:-2]):
#             layer = self.net_fwd.__getattr__(f'W{i+1}')
#             dct[f'W{i+1}'] = layer.weight
#         dct['U'] = self.net_fwd.U.weight

#         dct['U.T'] = self.net_bck.U.weight.T
#         for i in reversed(range(len(self.dims[1:-2]))):
#             layer = self.net_bck.__getattr__(f'W{i+1}')
#             dct[f'W{i+1}.T'] = layer.weight
#         dct['V.T'] = self.net_bck.V.weight.T

#         return dct


# class AEDenseVariational(nn.Module):

#     def __init__(self, dims: List[int], **kwargs) -> None:
#         super(AEDenseVariational, self).__init__()
#         assert len(dims) > 2, f'No. dimensions must be d > 2, found {len(dims)}.'
#         self.dims = dims
#         self.activation, self.activation_params = _set_default_activation(**kwargs)
#         layer = modules.AdjointDenseAffine
#         dct_fwd = OrderedDict()
#         dct_bck = OrderedDict()

#         # Construct layer for each dimension in forward model.
#         dct_fwd['V'] = layer(dims[0], dims[1], **kwargs)
#         dct_fwd['gamma0'] = self.activation(**self.activation_params)

#         for i, w in enumerate(zip(dims[1:-2], dims[2:-1])):
#             dct_fwd[f'W{i+1}'] = layer(w[0], w[1], **kwargs)
#             dct_fwd[f'gamma{i+1}'] = self.activation(**self.activation_params)

#         dct_fwd['U'] = modules.AdjointStochastic(dims[-2], dims[-1], **kwargs)

#         # Construct layer for each dimension in adjoint/inverse model.
#         # NOTE: These are added in the opposite "forward" direction, and we use
#         #       the adjoint method T(input) to call the network in this direction.
#         dct_bck['V'] = layer(dims[0], dims[1], **kwargs)
#         dct_bck['gamma0'] = self.activation(**self.activation_params)

#         for i, w in enumerate(zip(dims[1:-2], dims[2:-1])):
#             dct_bck[f'W{i+1}'] = layer(w[0], w[1], **kwargs)
#             dct_bck[f'gamma{i+1}'] = self.activation(**self.activation_params)

#         dct_bck['U'] = modules.AdjointStochastic(dims[-2], dims[-1], **kwargs)

#         self.net_fwd = modules.AdjointSequential(dct_fwd)
#         self.net_bck = modules.AdjointSequential(dct_bck)
#         self.forward = self.net_fwd.forward
#         self.T = self.net_bck.T


#     def get_weight_dict(self):
#         """Extracts a dictionary with relevant weight matrices.
#         """
#         dct = OrderedDict()
#         dct['V'] = self.net_fwd.V.weight
#         for i, _ in enumerate(self.dims[1:-2]):
#             layer = self.net_fwd.__getattr__(f'W{i+1}')
#             dct[f'W{i+1}'] = layer.weight
#         dct['U'] = self.net_fwd.U.weight

#         dct['U.T'] = self.net_bck.U.weight.T
#         for i in reversed(range(len(self.dims[1:-2]))):
#             layer = self.net_bck.__getattr__(f'W{i+1}')
#             dct[f'W{i+1}.T'] = layer.weight
#         dct['V.T'] = self.net_bck.V.weight.T

#         return dct


# class AEDenseDiagonal(nn.Module):

#     def __init__(self, dims: List[int], **kwargs) -> None:
#         super(AEDenseDiagonal, self).__init__()
#         assert len(dims) > 2, f'No. dimensions must be d > 2, found {len(dims)}.'
#         self.dims = dims
#         self.activation, self.activation_params = _set_default_activation(**kwargs)
#         layer = modules.AdjointDenseAffine
#         dct_fwd = OrderedDict()
#         dct_bck = OrderedDict()

#         # Construct layer for each dimension in forward model.
#         dct_fwd['V'] = layer(dims[0], dims[1], **kwargs)
#         dct_fwd['gamma0'] = self.activation(**self.activation_params)

#         for i, w in enumerate(zip(dims[1:-2], dims[2:-1])):
#             dct_fwd[f'W{i+1}'] = layer(w[0], w[1], **kwargs)
#             dct_fwd[f'gamma{i+1}'] = self.activation(**self.activation_params)

#         dct_fwd['S'] = modules.AdjointDiagonal(dims[-2], **kwargs)
#         dct_fwd['U'] = layer(dims[-2], dims[-1], **kwargs)

#         # Construct layer for each dimension in adjoint/inverse model.
#         # NOTE: These are added in the opposite "forward" direction, and we use
#         #       the adjoint method T(input) to call the network in this direction.
#         dct_bck['V'] = layer(dims[0], dims[1], **kwargs)
#         dct_bck['gamma0'] = self.activation(**self.activation_params)

#         for i, w in enumerate(zip(dims[1:-2], dims[2:-1])):
#             dct_bck[f'W{i+1}'] = layer(w[0], w[1], **kwargs)
#             dct_bck[f'gamma{i+1}'] = self.activation(**self.activation_params)

#         dct_bck['S'] = modules.AdjointDiagonal(dims[-2], **kwargs)
#         dct_bck['U'] = layer(dims[-2], dims[-1], **kwargs)

#         self.net_fwd = modules.AdjointSequential(dct_fwd)
#         self.net_bck = modules.AdjointSequential(dct_bck)

#         self.forward = self.net_fwd.forward
#         self.T = self.net_bck.T


#     def get_weight_dict(self):
#         """Extracts a dictionary with relevant weight matrices.
#         """
#         dct = OrderedDict()
#         dct['V'] = self.net_fwd.V.weight
#         for i, _ in enumerate(self.dims[1:-2]):
#             layer = self.net_fwd.__getattr__(f'W{i+1}')
#             dct[f'W{i+1}'] = layer.weight
#         dct['U'] = self.net_fwd.U.weight

#         dct['U.T'] = self.net_bck.U.weight.T
#         for i in reversed(range(len(self.dims[1:-2]))):
#             layer = self.net_bck.__getattr__(f'W{i+1}')
#             dct[f'W{i+1}.T'] = layer.weight
#         dct['V.T'] = self.net_bck.V.weight.T

#         return dct
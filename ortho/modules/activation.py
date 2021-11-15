from .. import (
    torch, np, opt, nn, F
)

from .adjoint import GaussianTruncator

# TODO: Implement parameterized versions.
# NOTE: No support for complex numbers yet. Possibly add modRelU approach?

class BiReLU(nn.Module):
    '''Bi-Lipschitzian (Leaky) Relu Activation.

    Class for an invertible leaky relu activation function. This class takes a single
    parameter for slope and lets the leaky slope be 1/slope in the forward direction.
    In the inverse direction, the slopes are reversed. This makes the activation
    invertible on R/C. The function is thus bi-Lipschitz with Lipschitz constant k.

    Parameters
    ----------
    k : int
        Slope for the forward activation on the positive number line.

    Attributes
    ----------
    k : int
        The slope for the positive number line, and the inverse of the negative number line.
    '''

    def __init__(self, k: float = 2):
        super(BiReLU, self).__init__()
        self.k = k

    def forward(self, input, *args, **kwargs):
        return input * (input >= 0).float() * self.k + input * (input < 0).float() / self.k

    def T(self, input, *args, **kwargs):
        return input * (input >= 0).float() / self.k + input * (input < 0).float() * self.k


class BiELU(nn.Module):

    '''Bi-Lipschitzian Exponential Linear Unit activation.

    Class for invertible leaky ELU activation. Similar to the invertible leaky ReLU
    activation function, but acts as the exponential linear unit in an interval between
    [lf, 0] where lf = df/dx^-1(1/k), i.e. a knot at the point where the derivative
    matches the inverse Lipschitz constant. The knots are calculated at initialization.

    Parameters
    ----------
    k : int
        Slope for the forward activation on the positive number line.

    Attributes
    ----------
    k : int
        The slope for the positive number line, and the inverse of the negative number line.
    '''

    def __init__(self, k: float = 2):
        super(BiELU, self).__init__()
        self.k = torch.tensor(k)
        self._set_knots()

    def _f(self, x):
        return self.k * (torch.exp(x) - 1)

    def _g(self, x):
        return  torch.log(x / self.k + 1)

    def _fdinv(self, x):
        return torch.log(x / self.k)

    def _gdinv(self, x):
        return 1 / x - self.k

    def _set_knots(self):
        self.lf = self._fdinv(1 / self.k)
        self.lg = self._gdinv(self.k)

    def forward(self, input, *args, **kwargs):
        pos = (input >= 0) * input * self.k
        neg = (input <= self.lf) * ((input - self.lf) / self.k + self.lg)
        splice = self._f(input * ((input < 0) & (input > self.lf)))
        return pos + neg + splice

    def T(self, input, *args, **kwargs):
        pos = (input >= 0) * input / self.k
        neg = (input <= self.lg) * ((input - self.lg) * self.k + self.lf)
        splice = self._g(input * ((input < 0) & (input > self.lg)))
        return pos + neg + splice


class BiCELU(nn.Module):

    '''Bi-Lipschitzian Continuously Differentiable Exponential Linear Unit activation.

    Class for invertible leaky CELU activation. Similar to ILELU but uses 4 knots. The
    extra knots from BiELU are simply given by the Lipschitz constant and 1/lf. As these
    are easily calculated, the Lipschitz constant can be a trainable parameter.

    Parameters
    ----------
    k : float
        Maximum slope and Lipshitz constant.

    beta : float
        Optional shape parameter.

    Attributes
    ----------
    k : float
        Maximum slope and Lipshitz constant.

    beta : float
        Optional shape parameter.
    '''

    def __init__(self, k: float = 2, beta: float = 0.5):
        super().__init__()
        self.k = torch.tensor(float(k))
        self.beta = torch.tensor(float(beta))
        self._set_knots()

    def _f(self, x):
        return self.beta * (torch.exp(x/self.beta) - 1)

    def _g(self, x):
        return self.beta * (torch.log(x/self.beta + 1))

    def _fdinv(self, x):
        return self.beta * torch.log(x)

    def _gdinv(self, x):
        return self.beta * (1/x - 1)

    def _set_knots(self):
        self.af = self._fdinv(1/self.k)
        self.bf = -self.af
        self.ag = self._gdinv(self.k)
        self.bg = self._gdinv(1/self.k)

    def forward(self, input, *args, **kwargs):
        pos = (input >= self.bf) * ((input - self.bf) * self.k + self.bg)
        neg = (input <= self.af) * ((input - self.af) / self.k + self.ag)
        splice = self._f(input * ((input < self.bf) & (input > self.af)))
        return pos + neg + splice

    def T(self, input, *args, **kwargs):
        pos = (input >= self.bg) * ((input - self.bg) / self.k + self.bf)
        neg = (input <= self.ag) * ((input - self.ag) * self.k + self.af)
        splice = self._g(input * ((input < self.bg) & (input > self.ag)))
        return pos + neg + splice

class BiSoft(nn.Module):

    '''Bi-Lipschitzian Softplus function.

    Class for invertible Softplus activation.

    Parameters
    ----------
    k : float
        Maximum slope and Lipshitz constant.

    beta : float
        Softness parameter

    Attributes
    ----------
    k : float
        Maximum slope and Lipshitz constant.

    beta : float
        Softness parameter
    '''

    def __init__(self, k: float = 2, beta: float = 1):
        super(BiSoft, self).__init__()
        self.k = torch.tensor(float(k))
        self.beta = torch.tensor(float(beta))
        self._set_knots()

    def _f(self, x):
        return self.k / self.beta * torch.log((1 + torch.exp(self.beta * x)) / 2)

    def _g(self, x):
        return torch.log(2*torch.exp(self.beta / self.k * x) - 1) / self.beta

    def _fdinv(self, x):
        return -torch.log(self.k / x - 1) / self.beta

    def _gdinv(self, x):
        return self.k / self.beta * torch.log(2 - 2 / (self.k * x))

    def _set_knots(self):
        self.af = self._fdinv(1/self.k)
        self.ag = self._gdinv(self.k)
        self.cf = self._f(self.af)
        self.cg = self._g(self.ag)

    def forward(self, input, *args, **kwargs):
        pos = (input >= self.af) * self._f(input)
        neg = (input < self.af) * ((input - self.af) / self.k + self.cf)
        return pos + neg

    def T(self, input, *args, **kwargs):
        pos = (input >= self.ag) * self._g(input)
        neg = (input < self.ag) * ((input - self.ag) / self.k + self.cg)
        return pos + neg


class InvertibleID(nn.Module):
    '''Invertible Identity activation.

    This is a generalized identity which can act in place of other activations in
    invertible neural networks.
    '''

    def __init__(self):
        super(InvertibleID, self).__init__()

    def forward(self, input, *args, **kwargs):
        '''Forward operation.
        '''
        return input

    def T(self, input, *args, **kwargs):
        '''Inverse operation.
        '''
        return input


class ReLU(nn.Module):
    '''Invertible Identity activation.

    This is a generalized identity which can act in place of other activations in
    invertible neural networks.
    '''

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, input, *args, **kwargs):
        '''Forward operation.
        '''
        return F.relu(input)

    def T(self, input, *args, **kwargs):
        '''Inverse operation.
        '''
        return F.relu(input)


class InvertibleSoftmax(nn.Module):
    '''Simple form of Invertible Softmax, using averaged logits.

    TODO: Fix this properly, no sigmoid, exp, finito!
    TODO: Docstring and documentation.
    '''

    def __init__(self):
        super(InvertibleSoftmax, self).__init__()


    def forward(self, input, get_last: bool = True):
        '''Forward operation.
        '''
        _z = torch.sigmoid(input)
        S = torch.sum(_z, -1)
        z = _z / S[...,None]
        if get_last:
            return z, S
        return z[...,:-1], S


    def T(self, input, S):
        '''Inverse operation.
        '''
        # TODO: Implement sampling / from_last flag
        sum = input[...,-1]
        input = self.get_last(input)
        return torch.log(input * sum[...,None])





class DirichletSoftmax(nn.Module):
    """Symmetric Fenton-Wilkinson-Wise Dirichlet-Softmax Transform.

    TODO: Docstring and documentation.
    """

    def __init__(
        self, features: int, alpha: float = 1.0, theta: float = 0.32,
        theta0: float = 1e-3, **kwargs
    ) -> None:

        super().__init__()
        self.features = features
        self.alpha = alpha
        self.init_params(alpha, theta, theta0, **kwargs)

    @staticmethod
    def cbrt(x):
        '''Helper method for robust cube roots.

        Parameters
        ----------
        x : torch.tensor
            Input tensor

        Returns
        -------
        Cube root of tensor.
        '''
        if torch.is_tensor(x):
            return torch.sign(x) * torch.abs(x)**(1/3)
        else:
            return np.cbrt(x)

    @staticmethod
    def beta2dirichlet(U):
        '''Performs the Beta->Dirichlet transform.

        Parameters
        ----------
        U : torch.tensor
            Tensor of Beta distributed variables.

        Returns
        -------
        Tensor of Dirichlet distributed variables.
        '''
        Y = torch.zeros_like(U)
        Y[...,0] = U[...,0]
        Y[...,1:] = torch.cumprod(1 - U[...,:-1], axis=-1) * U[...,1:]
        return Y

    @staticmethod
    def dirichlet2beta(Y):
        '''Perform the Dirichlet->Beta transform.

        Parameters
        ----------
        Y : torch.tensor
            Tensor of Dirichlet distributed variables.

        Returns
        -------
        Tensor of Beta distributed variables.

        '''
        U = torch.zeros_like(Y)
        U[...,0] = Y[...,0]
        U[...,1:] = Y[...,1:] / (1 - torch.cumsum(Y[...,:-1], axis=-1))
        return U

    @staticmethod
    def get_last_symdirichlet(Y):
        '''Retrieves last sample of a Symmetric Dirichlet and concatenates the result.

        Parameters
        ----------
        Z : torch.tensor
            Set of Gaussian random variables.

        clip : bool
            Flag to override default clip behaviour.
            See class docstring for details.

        Returns
        -------
        A tuple Y, S of Dirichlet distributed variables (Y) and normalizing constants (S)

        '''
        Ylast = 1 - torch.sum(Y, axis=-1)
        return torch.hstack([Y, Ylast[...,None]])

    def init_params(self, alpha: float, theta: float, theta0: float, **kwargs) -> None:
        """Initialize parameters for transform.

        Calculates parameters for both Fenton-Wilkinson and Wise-Transform and stores
        the parameters as buffers in the module.

        Parameters
        ----------
        alpha : float
            Parameter for symmetric Dirichlet distribution.

        theta : float
            Scale parameter for normalization of std. deviation of Wise-Transform.

        theta0 : float
            Translation parameter for normalization of std. deviation of Wise-Transform.
        """

        if 'device' in kwargs:
            device = kwargs['device']
        else:
            device = 'cpu'

        # Generate tensor-floats for computation
        one = torch.tensor(1, dtype=torch.double)
        k = torch.tensor(self.features, dtype=torch.double)

        # Calculate Fenton Wilkinson parameters
        sd_S = torch.sqrt(torch.log((torch.exp(one) - 1) / k + 1))

        # Calculate Wise parameters
        ks = torch.arange(k, 1, -1, dtype=torch.double)
        N = (ks - 1) * alpha + alpha / 2 - 1 / 2
        mu_Z = self.cbrt(alpha) * (1 - 1/(9 * alpha)) / \
            self.cbrt(N - (alpha - 1) * (alpha + 1/3) / (12 * N**2))
        sd_Z = theta / (alpha**(1/6) * self.cbrt(N)) + theta0

        # Vectorize parameters
        mu_Z = mu_Z[None, ...]
        sd_Z = sd_Z[None, ...]

        # Register parameters as buffers
        self.register_buffer('sd_S', sd_S.to(device))
        self.register_buffer('mu_Z', mu_Z.to(device))
        self.register_buffer('sd_Z', sd_Z.to(device))
        self.register_buffer('N', N.to(device))


    def gauss2beta(self, Z):
        '''Transforms Gaussians to Betas.

        Parameters
        ----------
        Z : torch.tensor
            Set of Gaussian random variables.

        clip : bool
            Flag to override default clip behaviour.
            See class docstring for details.

        Returns
        -------
        A tuple U, S of Beta distributed variables (U) and a normalizing constant (S)

        '''
        z  = Z[...,:-1]
        z0 = Z[...,-1][...,None]
        S = z0 * self.sd_S

        wise_normal = torch.clip(self.sd_Z * (z - S) + self.mu_Z, min=0)
        U = 1 - torch.exp(-wise_normal**3)
        return (U, S)

    def beta2gauss(self, U, S=None):
        '''Transforms Betas to Gaussians.

        Parameters
        ----------
        U : torch.tensor
            Set of Beta distributed random variables.

        S : torch.tensor
            Set of normal variables acting as the denominator of the softmax.

        Returns
        -------
        Set of standard Gaussian rvs.

        '''
        device = self.sd_S.device
        if S is None:
            S = torch.randn(len(U), 1).to(device) * self.sd_S

        z = (self.cbrt(-torch.log(1 - U.clip(0,1-3.5e-8))) - self.mu_Z) / self.sd_Z + S
        z0 = S / self.sd_S
        return torch.hstack([z, z0])

    def forward(self, Z, *args, convert_dtype=True, get_last=True, **kwargs):
        Z = Z.double()
        U, S = self.gauss2beta(Z)
        Y = self.beta2dirichlet(U)
        if convert_dtype:
            Y, S = Y.float(), S.float()
        if not get_last:
            return Y.clip(0,1), S
        else:
            return self.get_last_symdirichlet(Y).clip(0,1), S

    def T(self, Y, S, *args, convert_dtype=True, get_last=True, **kwargs):
        '''Performs the inverse Dirichlet-Softmax Transform.

        Parameters
        ----------
        Y : torch.tensor
            Set of Dirichlet distributed random variables.

        S : torch.tensor
            Set of normal variables acting as the denominator of the softmax.

        Returns
        -------
        Set of standard Gaussian rvs, transformed from Dirichlet.
        '''
        if get_last:
            Y = Y[...,:-1]
        if convert_dtype:
            Y = Y.double()
            if S is not None:
                S = S.double()
        U = self.dirichlet2beta(Y)
        Z = self.beta2gauss(U, S)
        Z = Z.float()
        return Z


class ConcurrentActivationWrapper(nn.Module):

    '''Wrapper for activations in concurrent sequential networks.
    '''

    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def forward(self, input_a, input_b, *args, **kwargs):
        return self.activation(input_a), self.activation(input_b)

    def T(self, input_a, input_b, *args, **kwargs):
        return self.activation.T(input_a), self.activation.T(input_b)

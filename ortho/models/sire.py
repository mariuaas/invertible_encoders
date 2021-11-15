from .. import (
    np, nn, torch, F, fft, modules, OrderedDict, List
)

from ..modules import AdjointSemiUnitary, BiCELU

class QuickSIRE(nn.Module):

    def __init__(self, size, activation=nn.ReLU(), **kwargs):
        super().__init__()
        self.size = size
        self.features = np.prod(size)
        self.no_nets = 1
        self._kwargs = kwargs
        self.activation = activation
        self.psis = nn.ModuleList([self.__conv(set_device=False)])
        self.phis = nn.ModuleList([self.__conv(set_device=False)])

    def __dense(self, set_device=True):
        net = nn.Sequential(
            nn.Flatten(-len(self.size), -1),
            AdjointSemiUnitary(self.features, self.features, **self._kwargs),
            self.activation,
            AdjointSemiUnitary(self.features, self.features, **self._kwargs),
            self.activation,
            AdjointSemiUnitary(self.features, self.features, **self._kwargs),
            nn.Unflatten(-1, [*self.size]),
        )
        if set_device:
            return net.to(next(self.parameters()).device)
        else:
            return net

    def __conv(self, set_device=True):
        net = nn.Sequential(
            nn.Conv2d(1, 3, 3, 1, 1, padding_mode='circular'),
            self.activation,
            nn.Conv2d(3, 5, 5, 1, 2, padding_mode='circular'),
            self.activation,
            nn.Conv2d(5, 3, 7, 1, 3, padding_mode='circular'),
            self.activation,
            nn.Conv2d(3, 1, 9, 1, 4, padding_mode='circular'),
        )
        if set_device:
            return net.to(next(self.parameters()).device)
        else:
            return net

    def add_dense(self):
        self.phis.append(self.__dense())
        self.psis.append(self.__dense())
        self.no_nets += 1


    def add_conv(self):
        self.phis.append(self.__conv())
        self.psis.append(self.__conv())
        self.no_nets += 1

    def phi_psi(self, x, y, k=0.4):
        for i in range(self.no_nets):
            x = x - k*self.psis[i](x)
            y = y - k*self.phis[i](y)
        return x, y

    def forward(self, x, y0=None, iter=5):
        if y0 is None:
            y0 = torch.ones_like(x) * 0.5
        for _ in range(iter):
            px, py = self.phi_psi(x, y0)
            y0 = y0 + px - py
        return y0

    def T(self, y, x0=None, iter=5):
        if x0 is None:
            x0 = torch.ones_like(y) * 0.5
        for _ in range(iter):
            px, py = self.phi_psi(x0, y)
            x0 = x0 + py - px
        return x0
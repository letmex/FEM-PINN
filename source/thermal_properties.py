import torch


class ThermalProperties:
    """Container for thermal material constants."""

    def __init__(self, alpha, rho, k0, c, T0, TFinal, Tref, thk):
        self.alpha = alpha
        self.rho = rho
        self.k0 = k0
        self.c = c
        self.T0 = T0
        self.TFinal = TFinal
        self.Tref = Tref
        self.thk = thk

    def to(self, device):
        self.alpha = self.alpha.to(device)
        self.rho = self.rho.to(device)
        self.k0 = self.k0.to(device)
        self.c = self.c.to(device)
        self.T0 = self.T0.to(device)
        self.TFinal = self.TFinal.to(device)
        self.Tref = self.Tref.to(device)
        self.thk = self.thk.to(device)
        return self

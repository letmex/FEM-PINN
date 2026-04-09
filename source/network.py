import warnings
import numpy as np
import torch
import torch.nn as nn


class SteepTanh(nn.Module):
    def __init__(self, coeff):
        super(SteepTanh, self).__init__()
        self.coeff = coeff

    def forward(self, x):
        return torch.tanh(self.coeff * x)


class SteepReLU(nn.Module):
    def __init__(self, coeff):
        super(SteepReLU, self).__init__()
        self.coeff = coeff

    def forward(self, x):
        return torch.relu(self.coeff * x)


class TrainableTanh(nn.Module):
    def __init__(self, init_coeff):
        super(TrainableTanh, self).__init__()
        self.coeff = nn.Parameter(torch.tensor(float(init_coeff)))

    def forward(self, x):
        return torch.tanh(self.coeff * x)


class TrainableReLU(nn.Module):
    def __init__(self, init_coeff):
        super(TrainableReLU, self).__init__()
        self.coeff = nn.Parameter(torch.tensor(float(init_coeff)))

    def forward(self, x):
        return torch.relu(self.coeff * x)


class AdaptiveReLU(nn.Module):
    """
    ReLU with learnable positive slope m:
      y = ReLU(m * x), m = softplus(m_raw) + 1e-6
    """

    def __init__(self, init_coeff=1.0):
        super(AdaptiveReLU, self).__init__()
        self.m_raw = nn.Parameter(torch.tensor(float(init_coeff)))

    def forward(self, x):
        m = torch.nn.functional.softplus(self.m_raw) + 1e-6
        return torch.relu(m * x)


class InputNormalization(nn.Module):
    """
    Normalize geometric input x to [0, 1] with domain extrema.
    """

    def __init__(self, domain_extrema=None):
        super(InputNormalization, self).__init__()
        if domain_extrema is None:
            self.register_buffer("x_min", torch.tensor([], dtype=torch.float32))
            self.register_buffer("x_scale", torch.tensor([], dtype=torch.float32))
            self.enabled = False
        else:
            dom = domain_extrema.detach().clone().to(torch.float32)
            x_min = dom[:, 0]
            x_max = dom[:, 1]
            x_scale = torch.clamp(
                x_max - x_min,
                min=torch.tensor(1e-12, dtype=dom.dtype, device=dom.device),
            )
            self.register_buffer("x_min", x_min)
            self.register_buffer("x_scale", x_scale)
            self.enabled = True

    def forward(self, x):
        if not self.enabled:
            return x
        x_min = self.x_min.to(device=x.device, dtype=x.dtype)
        x_scale = self.x_scale.to(device=x.device, dtype=x.dtype)
        return (x - x_min) / x_scale


def activations(activation, init_coeff, n_hidden_layers=1):
    if activation == "SteepTanh":
        acts = SteepTanh(init_coeff)
        trainable = False
    elif activation == "SteepReLU":
        acts = SteepReLU(init_coeff)
        trainable = False
    elif activation == "TrainableTanh":
        acts = nn.ModuleList([TrainableTanh(init_coeff) for _ in range(n_hidden_layers)])
        trainable = True
    elif activation in ("TrainableReLU", "AdaptiveReLU"):
        act_cls = TrainableReLU if activation == "TrainableReLU" else AdaptiveReLU
        acts = nn.ModuleList([act_cls(init_coeff) for _ in range(n_hidden_layers)])
        trainable = True
    else:
        warnings.warn(
            "Prescribed activation does not match available choices. Using default Tanh activation."
        )
        acts = nn.Tanh()
        trainable = False
    return acts, trainable


def init_xavier(model):
    activation = getattr(model, "name_activation", "tanh")
    init_coeff = float(getattr(model, "init_coeff", 1.0))

    if activation in ("TrainableReLU", "SteepReLU") and init_coeff < 1.0:
        raise ValueError(
            f"Invalid init_coeff={init_coeff} for {activation}. "
            "ReLU-family Xavier gain in this project requires init_coeff >= 1.0."
        )

    if activation in ("TrainableTanh", "SteepTanh"):
        gain = nn.init.calculate_gain("tanh") / max(init_coeff, 1e-12)
    elif activation in ("TrainableReLU", "SteepReLU"):
        slope = np.sqrt(max(init_coeff**2 - 1.0, 0.0))
        gain = nn.init.calculate_gain("leaky_relu", slope)
    elif activation == "AdaptiveReLU":
        gain = nn.init.calculate_gain("relu")
    else:
        gain = nn.init.calculate_gain("tanh")

    def _init(m):
        if isinstance(m, nn.Linear) and m.weight.requires_grad:
            nn.init.xavier_uniform_(m.weight, gain=gain)
            if m.bias is not None and m.bias.requires_grad:
                m.bias.data.fill_(0.0)

    model.apply(_init)


class NeuralNet(nn.Module):
    """
    Generic MLP kept for legacy non-TM workflows.
    """

    def __init__(self, input_dimension, output_dimension, n_hidden_layers, neurons, activation, init_coeff=1.0):
        super(NeuralNet, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.neurons = neurons
        self.n_hidden_layers = n_hidden_layers
        self.name_activation = activation
        self.init_coeff = init_coeff
        self.trainable_activation = False

        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers - 1)]
        )
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)
        self.activations, self.trainable_activation = activations(
            activation=activation,
            init_coeff=init_coeff,
            n_hidden_layers=n_hidden_layers,
        )

    def forward(self, x):
        if self.trainable_activation:
            x = self.activations[0](self.input_layer(x))
            for j, layer in enumerate(self.hidden_layers):
                x = self.activations[j + 1](layer(x))
            return self.output_layer(x)
        x = self.activations(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activations(layer(x))
        return self.output_layer(x)


class MonolithicTMPhaseNet(nn.Module):
    """
    Monolithic thermo-mechanical-phase network.
      input: [x, y]
      output: [T_raw, ux_raw, uy_raw, d_raw]

    No trunk/head split and no branch-parameter interfaces.
    """

    def __init__(
        self,
        spatial_dimension,
        n_hidden_layers,
        neurons,
        activation,
        init_coeff=1.0,
        seed=1,
        phase_output_bias_init=-4.0,
        domain_extrema=None,
        use_input_normalization=True,
    ):
        super(MonolithicTMPhaseNet, self).__init__()
        self.input_dimension = spatial_dimension
        self.n_hidden_layers = n_hidden_layers
        self.neurons = neurons
        self.name_activation = activation
        self.init_coeff = init_coeff
        self.trainable_activation = False

        if use_input_normalization and domain_extrema is not None:
            self.input_norm = InputNormalization(domain_extrema=domain_extrema)
        else:
            self.input_norm = InputNormalization(domain_extrema=None)

        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers - 1)]
        )
        self.output_layer = nn.Linear(self.neurons, 4)
        self.activations, self.trainable_activation = activations(
            activation=activation,
            init_coeff=init_coeff,
            n_hidden_layers=n_hidden_layers,
        )

        torch.manual_seed(seed)
        init_xavier(self)
        with torch.no_grad():
            # output order: [T_raw, ux_raw, uy_raw, d_raw]
            self.output_layer.bias[3] = float(phase_output_bias_init)

    def forward(self, x):
        x = self.input_norm(x)
        if self.trainable_activation:
            h = self.activations[0](self.input_layer(x))
            for j, layer in enumerate(self.hidden_layers):
                h = self.activations[j + 1](layer(h))
            return self.output_layer(h)
        h = self.activations(self.input_layer(x))
        for layer in self.hidden_layers:
            h = self.activations(layer(h))
        return self.output_layer(h)

    def forward_raw(self, x):
        out = self.forward(x)
        return out[:, 0], out[:, 1], out[:, 2], out[:, 3]


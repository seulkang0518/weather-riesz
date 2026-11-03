import torch
from torch import nn


class TemperatureCGM(nn.Module):
    """Port of mvpp/cgm_models_linear.py::_build_model_t2m.

    Inputs [B,D,17], [B,D,17], [B,D,39]; outputs [B,M,D].
    delta is a *multiplier*, exactly as in code (not sqrt(delta)).
    """
    def __init__(self, stations=10, latent_dim=10):
        super().__init__()
        self.stations, self.latent_dim = stations, latent_dim
        self.mean_weight = nn.Parameter(torch.empty(stations, 17))
        self.mean_bias = nn.Parameter(torch.zeros(stations))
        self.noise_scale = nn.Linear(stations * 17, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(stations * 39 + latent_dim, 100), nn.ELU(),
                                     nn.Linear(100, 100), nn.ELU(), nn.Linear(100, stations))
        nn.init.xavier_uniform_(self.mean_weight)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, mean_inputs, std_inputs, all_inputs, z):
        # z is explicitly supplied: reproducible paired sampling across objectives.
        b, m, _ = z.shape
        mean = (mean_inputs * self.mean_weight).sum(-1) + self.mean_bias
        delta = self.noise_scale(std_inputs.flatten(1)).exp()
        context = all_inputs.flatten(1)[:, None].expand(b, m, -1)
        residual = self.decoder(torch.cat([context, z * delta[:, None]], -1))
        return mean[:, None] + residual

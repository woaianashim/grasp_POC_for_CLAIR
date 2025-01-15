import jax.numpy as np
import flax.linen as nn
from .layers import LinNormAct
from typing import List, Dict

__all__ = ["MLPWithGRU", "MLPHead"]


class MLPWithGRU(nn.Module):
    hidden_sizes: List[int]
    layers_cfg: dict

    @nn.compact
    def __call__(self, x, h):
        # Intermediate MLP layers
        for features in self.hidden_sizes[:-1]:
            x = LinNormAct(**self.layers_cfg, features=features)(x)

        # GRU layer
        h, x = nn.GRUCell(features=self.hidden_sizes[-1])(h, x)
        out = x
        return out, h

    def initialize_hidden(self, batch):
        batch_size = batch.shape[1]
        return np.zeros((batch_size, self.hidden_sizes[-1]))


class MLPHead(nn.Module):
    hidden_sizes: List[int]
    linear_kwargs: Dict
    out_kwargs: Dict
    out_features: int
    skip_connection: bool = False

    def setup(self):
        self.layers = [
            LinNormAct(**self.linear_kwargs, features=features)
            for features in self.hidden_sizes
        ]
        if self.skip_connection:
            self.skip_layer = LinNormAct(
                **self.linear_kwargs, features=self.hidden_sizes[-1]
            )
        self.out_layer = LinNormAct(features=self.out_features, **self.out_kwargs)

    def __call__(self, inp):
        x = inp
        for layer in self.layers:
            x = layer(x)
        if self.skip_connection:
            x = self.skip_layer(inp)
        return self.out_layer(x)


class ActorHead(MLPHead):
    skip_connection: bool = True

    def __call__(self, x):
        mean_logstd = super().__call__(x)
        mean, log_std = np.split(mean_logstd, 2, axis=-1)
        return mean, log_std

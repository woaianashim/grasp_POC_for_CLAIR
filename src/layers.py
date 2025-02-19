from jax.nn.initializers import variance_scaling
import jax.numpy as np
import flax.linen as nn

dtype = np.float32  # np.bfloat16


class BaseModule(nn.Module):
    def get_layer_norm(self, norm_layer):
        if norm_layer == "LayerNorm":
            return nn.LayerNorm(dtype=self.dtype)
        return lambda x: x

    def get_activation(self, act_fn):
        if act_fn == "relu":
            return nn.relu
        if act_fn == "tanh":
            return nn.tanh
        if act_fn == "silu":
            return nn.silu
        if act_fn == "gelu":
            return nn.gelu
        return lambda x: x


class LayerNormAct(BaseModule):
    act_fn: str
    norm_layer: str
    init_kwargs: dict

    def setup(self):
        self.dtype = dtype  # FIX: bfloat16 failed for some reason (~Wx=Wy)
        self.norm = self.get_layer_norm(self.norm_layer)
        self.act = self.get_activation(self.act_fn)

    def __call__(
        self,
        x,
    ):
        x = self.layer(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class LinNormAct(LayerNormAct):
    features: int
    use_bias: bool

    def setup(self):
        super().setup()
        self.layer = nn.Dense(
            features=self.features,
            use_bias=self.use_bias,
            kernel_init=variance_scaling(**self.init_kwargs),
            # dtype=self.dtype,
        )


class ConvNormAct(LayerNormAct):
    features: int
    kernel_size: tuple
    strides: tuple
    padding: str
    use_bias: bool
    transpose: bool = False

    def setup(self):
        super().setup()
        conv = nn.Conv if not self.transpose else nn.ConvTranspose
        self.layer = conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            kernel_init=variance_scaling(**self.init_kwargs),
            padding=self.padding,
            use_bias=self.use_bias,
            # dtype=self.dtype,
        )

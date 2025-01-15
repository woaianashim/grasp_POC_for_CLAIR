import distrax
import jax
import jax.numpy as np
from functools import partial


def cast_to_type(x, dtype):
    return np.astype(x, dtype)


cast = partial(cast_to_type, dtype=np.float32)
# cast = partial(cast_to_type, dtype=np.bfloat16) FIX still same bf16 problem
uncast = partial(cast_to_type, dtype=np.float32)


def cast_fn(fun):  # Yeah, right now it is useless
    def wrapped(self, *args, **kwargs):
        args = [uncast(arg) for arg in args]
        res = fun(self, *args, **kwargs)
        return cast(res)

    return wrapped


class C51(distrax.Categorical):  # C51 with symlog transform
    def __init__(
        self,
        logits,
        low=-20.0,
        high=20.0,
    ):
        super().__init__(logits=uncast(logits))
        self.bins = np.linspace(low, high, num=logits.shape[-1])

    def mean(self):
        tmean = np.sum(self.bins * self.probs, axis=-1, keepdims=True)
        return symexp(tmean)

    @cast_fn
    def log_prob(self, value):
        tvalue = symlog(value)
        n_bins = len(self.bins)
        step = self.bins[1] - self.bins[0]

        floor_ind = np.clip((tvalue >= self.bins).sum(-1) - 1, 0, n_bins - 1)
        ohc_floor = jax.nn.one_hot(floor_ind, n_bins)
        ceil_ind = np.clip(n_bins - (tvalue < self.bins).sum(-1), 0, n_bins - 1)
        ohc_ceil = jax.nn.one_hot(ceil_ind, n_bins)

        oob = floor_ind == ceil_ind  # value out of scale

        tvalue = tvalue.squeeze(-1)
        dist_to_floor = (
            np.abs(tvalue - self.bins[floor_ind]) * (1 - oob) + oob / 2
        )  # FIX not accurate with oob, but it appears for more than 1e8 reward only
        dist_to_ceil = np.abs(tvalue - self.bins[ceil_ind]) * (1 - oob) + oob / 2

        log_prob = (
            self.logits
            * (
                ohc_floor * dist_to_ceil[..., None]
                + ohc_ceil * dist_to_floor[..., None]
            )
            / step
        )
        return log_prob


def symlog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def symexp(x):
    return np.sign(x) * np.expm1(np.abs(x))


class Normal(distrax.Independent):
    def __init__(self, mean, log_std, std_scale=2e0):
        mean, log_std = uncast(mean), uncast(log_std)
        std = np.exp(log_std) * std_scale
        action_dist = distrax.Normal(mean, std)
        super().__init__(distribution=action_dist, reinterpreted_batch_ndims=1)

    def sample_and_log_prob(self, seed=None):
        sample, logprobs = super().sample_and_log_prob(seed=seed)
        return cast(sample), cast(logprobs)

    @cast_fn
    def log_prob(self, action):
        return super().log_prob(action)

import jax
import jax.numpy as np
from dataclasses import dataclass


@dataclass
class BaseTransition:
    _obses: dict = None
    _prev_obses: dict = None
    action: np.ndarray = None
    done: np.ndarray = False
    trunc: np.ndarray = False
    reward: np.ndarray = None
    action_logprob: np.ndarray = None
    value: np.ndarray = None
    value_logits: np.ndarray = None
    action_logstd: np.ndarray = None
    obs_keys: list = None

    @property
    def obs(self):  # Used in agent.step
        return np.concatenate([self._obses[key] for key in self.obs_keys], -1)

    @property
    def prev_obs(self):  # Used in agent.step
        return np.concatenate([self._prev_obses[key] for key in self.obs_keys], -1)

    def __getitem__(self, key):
        if key in self.obses:
            return self.obses[key]
        else:
            return getattr(self, key)

    @property
    def obses(self):  # Used in buffer with delay
        return self._prev_obses

    @obses.setter
    def obses(self, new_obses):
        if not self.trunc:
            self._prev_obses = self._obses
        self._obses = new_obses


class GPUPPOBuffer:
    def __init__(
        self,
        env,
        seq_len,
        n_covers,
        batch_size,
        value_logits_bins,
        gamma=0.97,
        gae_lambda=0.95,
        seed=1,
    ):
        obs_space, act_space = env.observation_space, env.action_space
        obs_shape = obs_space.shape
        act_shape = act_space.shape
        assert (
            obs_shape[0] % batch_size == 0
        ), "Batch size must divide number of environments"
        self.key = jax.random.PRNGKey(seed)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_covers = n_covers
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.step = 0

        def _zeros(shape=(), dtype=np.uint8):
            return np.zeros((seq_len,) + shape, dtype)

        self._transition = BaseTransition()
        self.buffer = {
            "action": _zeros(act_shape, act_space.dtype),
            "action_logprob": _zeros((act_shape[0],), act_space.dtype),
            "done": _zeros((obs_shape[0],)),
            "trunc": _zeros((obs_shape[0],)),
            "reward": _zeros((obs_shape[0], 1), dtype=np.float32),
            "value": _zeros((obs_shape[0], 1), dtype=np.float32),
            "value_logits": _zeros((obs_shape[0], value_logits_bins), dtype=np.float32),
            "action_logstd": _zeros(act_shape, act_space.dtype),
        }
        self.obs_keys = []
        for key in obs_space:
            self.obs_keys.append(key)
            key_space = obs_space[key]
            self.buffer[key] = _zeros(
                key_space.shape,
                key_space.dtype,
            )
        self._transition.obs_keys = self.obs_keys

    def reset(self, latent_state=None, obses=None, *args, **kwargs):
        self.latent_state = latent_state
        self.step = 0
        if "returns" in self.buffer:
            del self.buffer["returns"]

    def add(self, transition):
        assert not self.ready, "Buffer is full"
        for key in self.buffer:
            self.buffer[key] = self.buffer[key].at[self.step].set(transition[key])
        self.step += 1

    def __iter__(self):
        for _ in range(self.n_covers):
            permutation = jax.random.permutation(self.rng, self.buffer["obs"].shape[1])
            n_batches = self.buffer["obs"].shape[1] // self.batch_size
            for batch_idx in range(n_batches):
                idx = permutation[
                    batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
                ]
                batch = {
                    k: v[:, idx]
                    for k, v in self.buffer.items()
                    if k not in self.obs_keys
                }
                obses = {
                    k: v[:, idx] for k, v in self.buffer.items() if k in self.obs_keys
                }
                batch["obs"] = self.combine_obs(obses)
                batch["hidden"] = self.latent_state[idx]
                yield _, batch  # No priority replay now, so indeces not required

    def prepare_returns(self, next_values):
        returns = [next_values]
        values = np.concat([self.buffer["value"], next_values[None]], axis=0)
        discount = self.gamma * (1 - self.buffer["done"][..., None])
        gae_rewards = self.buffer["reward"] + discount * values[1:] * (
            1 - self.gae_lambda
        )
        for gae_reward, disc in zip(gae_rewards[::-1], discount[::-1]):
            returns.append(gae_reward + disc * returns[-1] * self.gae_lambda)
        self.buffer["returns"] = np.stack(returns[::-1], axis=0)[:-1]

    @property
    def dummy_batch(self):
        sample = {k: v[:, :1] for k, v in self.buffer.items() if k not in self.obs_keys}
        sample["obs"] = np.concatenate(
            [self.buffer[k][: self.seq_len + 1, :1] for k in self.obs_keys], -1
        )
        sample["returns"] = sample["value"]
        return sample

    @property
    def rng(self):
        self.key, subkey = jax.random.split(self.key)
        return subkey

    @property
    def ready(self):
        return self.step == self.seq_len

    @property
    def transition(self):
        return self._transition

    def combine_obs(self, obses):
        return np.concatenate([obses[k] for k in self.obs_keys], -1)

    def update(self, indeces, priorities):  # For compatibility
        pass

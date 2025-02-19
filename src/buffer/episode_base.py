import jax.numpy as np
import numpy as _np
from .priority_sampler import PrioritiesSampler, Mode
from dataclasses import dataclass

dtype = np.float32  # np.bfloat16


@dataclass
class BaseTransition:
    obses: dict = None
    action: np.ndarray = None
    done: np.ndarray = False
    trunc: np.ndarray = False
    reward: np.ndarray = None
    obs_keys: list = None

    @property
    def obs(self):
        return np.concatenate([self.obses[key] for key in self.obs_keys], -1)

    def __getitem__(self, key):
        if key in self.obses:
            return self.obses[key]
        else:
            return getattr(self, key)


class BaseEpisodeBuffer:
    def __init__(
        self,
        env,
        batch_size,
        seq_len,
        capacity,
        initial_size=2048,
        n_updates=1,
        seed=1,
        mode=Mode.uniform,
    ):
        obs_space, act_space = env.observation_space, env.action_space
        act_shape = act_space.shape
        self.n_envs = env.n_envs
        self.env = env.n_envs
        self.mode = mode
        self._transition = BaseTransition()

        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer = {
            "action": self._zeros(act_shape, act_space.dtype),
            "done": self._zeros((self.n_envs,)),
            "trunc": self._zeros((self.n_envs,)),
            "reward": self._zeros((self.n_envs, 1), dtype=_np.float32),
        }
        self.obs_keys = []
        for key in obs_space:
            self.obs_keys.append(key)
            key_space = obs_space[key]
            self.buffer[key] = self._zeros(
                key_space.shape, key_space.dtype, extra_capacity=1
            )
        self._transition.obs_keys = self.obs_keys
        self.seq_len = seq_len
        self.n_updates = n_updates
        self.initial_size = initial_size

    def add(self, transition, info=None):
        for key in self.buffer:
            self.buffer[key][
                (self.step + (1 if key in self.obs_keys else 0)) % self.capacity
            ] = transition[key]
        self.update_sampler()
        if self.buffer["trunc"][self.step, 0]:
            self.seq_start_step = (self.step + 1) % self.capacity
        elif self.step == 0:
            self.seq_start_step = 0
        else:
            self.seq_start_step = max(
                (self.step + 1) % self.capacity - self.seq_len, self.seq_start_step
            )
        self.step = (self.step + 1) % self.capacity
        if self.step == 0:
            self.sampler.count = 0
        # self.size = max(self.size, self.step)

    def update_sampler(self):  # FIX relies on synced truncation of env
        for i in range(self.n_envs):
            skip = (
                (self.step + 1) % self.capacity - self.seq_start_step
                < self.seq_len  # Too short since start of seq
            ) or (
                self.buffer["trunc"][self.step, i]  # if trunc, next obs is absent
                * (
                    1 - self.buffer["done"][self.step, i]
                )  # if done, previous is not matter
            )
            self.sampler.add((self.seq_start_step, i), skip=skip)  # Insert if OK
            self.sampler.remove((self.step, i))  # Del overriden
        self.size = min(
            self.size + 1 - skip, self.capacity
        )  # Not exact, due to old skips

    def update(self, indeces, priorities):
        priorities = _np.array(priorities)
        if self.mode != Mode.uniform:
            for index, priority in zip(indeces, priorities):
                self.sampler.update(index, priority)

    def __iter__(self):
        for _ in range(self.n_updates):
            yield self.sample()

    def sample(self):
        indeces = self.sampler.sample()
        batch, _, _, _ = self[indeces]
        return indeces, batch

    def __getitem__(self, indeces):
        indeces = _np.array(indeces).T  # 2 x B
        first_steps, envs = indeces
        steps = first_steps[None] + np.arange(self.seq_len)[:, None]  # CHECK TODO
        steps_ext = first_steps[None] + _np.arange(self.seq_len + 1)[:, None]
        inds = (steps, envs[None])
        inds_ext = (steps_ext, envs[None])
        batch = {k: self.buffer[k][inds] for k in self.buffer if k not in self.obs_keys}
        obses = {k: self.buffer[k][inds_ext] for k in self.obs_keys}
        batch["obs"] = self.combine_obs(obses)
        return batch, obses, inds, inds_ext

    def reset(self, obses=None, *args, **kwargs):
        self.step = 0
        self.seq_start_step = 0
        self.size = 0
        self.sampler = PrioritiesSampler(
            self.capacity * self.n_envs, self.batch_size, mode=self.mode
        )
        self.reset_obses(obses)

    def reset_obses(self, obses):
        for key in self.obs_keys:
            self.buffer[key][self.step] = obses[key]

    @property
    def dummy_batch(self):
        sample = {
            k: v[: self.seq_len, :1]
            for k, v in self.buffer.items()
            if k not in self.obs_keys
        }
        sample["obs"] = np.concatenate(
            [self.buffer[k][: self.seq_len + 1, :1] for k in self.obs_keys], -1
        )
        return sample

    @property
    def ready(self):
        return self.size > max(self.seq_len * 2, self.initial_size)

    def _zeros(self, shape=(), dtype=np.uint8, extra_capacity=0):
        return _np.zeros((self.capacity + extra_capacity,) + shape, dtype)

    @property
    def transition(self):
        return self._transition

    @property
    def next_step(self):
        return (self.step + 1) % self.capacity

    def combine_obs(self, obses):
        return np.concatenate([obses[k] for k in self.obs_keys], -1)

from .episode_base import BaseEpisodeBuffer
from .priority_sampler import Mode
import numpy as _np
from jax.tree_util import tree_map
from torch import from_numpy


class HEREpisodeBuffer(BaseEpisodeBuffer):  # Assuming goal is last 3 coordinates of obs
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
        her_ratio=0.6,
    ):
        super().__init__(
            env,
            batch_size,
            seq_len,
            capacity,
            initial_size=initial_size,
            n_updates=n_updates,
            mode=mode,
            seed=seed,
        )
        self.env = env
        self.her_ratio = her_ratio
        self.current_start = 0
        self.starts = self._zeros((self.n_envs,), dtype=_np.uint32)
        self.ends = self._zeros((self.n_envs,), dtype=_np.uint32)
        self.extra_info = {"overdraft": self._zeros((self.n_envs,), dtype=float)}

    def add(self, transition, info=None):
        self.starts[self.step] = self.current_start
        self.extra_info["overdraft"][self.next_step] = info["overdraft"]
        if transition["trunc"] or self.next_step == 0:
            assert self.step >= self.current_start
            self.ends[self.current_start : self.step + 1] = self.step
            self.current_start = self.next_step
        super().add(transition, info)

    def __getitem__(self, indeces):
        batch, obses, inds, inds_ext = super().__getitem__(indeces)
        indeces = _np.array(indeces).T  # 2 x B
        first_steps, envs = indeces
        starts = self.starts[first_steps, envs]
        ends = self.ends[first_steps, envs]
        ends = ends * (1 - (starts == self.current_start)) + (
            starts == self.current_start
        ) * (self.step + 1)
        goal_inds = _np.random.randint(first_steps, ends + 1)
        new_goal = self.buffer["goal"][goal_inds, envs]
        replace_inds = _np.random.choice(
            _np.arange(self.batch_size, dtype=int),
            size=int(self.batch_size * self.her_ratio),
            replace=False,
        )
        obses["goal"][:, replace_inds] = new_goal[replace_inds]
        torch_obses = tree_map(lambda x: from_numpy(x), obses)
        torch_obses["overdraft"] = from_numpy(self.extra_info["overdraft"][inds_ext])
        old_states = {k: obs[:-1] for k, obs in torch_obses.items()}
        states = {k: obs[1:] for k, obs in torch_obses.items()}
        new_rewards, _ = self.env.get_reward(states, old_states)
        batch["reward"] = new_rewards.numpy()[..., None]

        batch["obs"] = self.combine_obs(obses)

        return batch, obses, inds, inds_ext

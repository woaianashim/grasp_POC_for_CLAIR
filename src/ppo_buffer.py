import jax
import jax.numpy as np


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

        self.buffer = {
            "obs": _zeros(obs_shape, obs_space.dtype),
            "action": _zeros(act_shape, act_space.dtype),
            "action_logprob": _zeros((act_shape[0],), act_space.dtype),
            "done": _zeros((obs_shape[0],)),
            "trunc": _zeros((obs_shape[0],)),
            "reward": _zeros((obs_shape[0], 1), dtype=np.float32),
            "value": _zeros((obs_shape[0], 1), dtype=np.float32),
            "value_logits": _zeros((obs_shape[0], value_logits_bins), dtype=np.float32),
            "action_logstd": _zeros(act_shape, act_space.dtype),
        }

    def reset(self, initial_latent_state):
        self.latent_state = initial_latent_state
        self.step = 0
        if "returns" in self.buffer:
            del self.buffer["returns"]

    def add(self, states):
        assert not self.full, "Buffer is full"
        for key in self.buffer:
            self.buffer[key] = self.buffer[key].at[self.step].set(states[key])
        self.step += 1

    def __iter__(self):
        for _ in range(self.n_covers):
            permutation = jax.random.permutation(self.rng, self.buffer["obs"].shape[1])
            n_batches = self.buffer["obs"].shape[1] // self.batch_size
            for batch_idx in range(n_batches):
                idx = permutation[
                    batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
                ]
                batch = {k: v[:, idx] for k, v in self.buffer.items()}
                batch["hidden"] = self.latent_state[idx]
                yield batch

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
        sample = {k: v[:, :1] for k, v in self.buffer.items()}
        sample["returns"] = sample["value"]
        return sample

    @property
    def rng(self):
        self.key, subkey = jax.random.split(self.key)
        return subkey

    @property
    def full(self):
        return self.step == self.seq_len

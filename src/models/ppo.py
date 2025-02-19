import flax.linen as nn
from jax.lax import stop_gradient as sg
import jax.numpy as np
from src.nets import MLPWithGRU, MLPHead, ActorHead
from typing import Dict, Tuple
from src.dists import C51, Normal


class PPOGraspModel(nn.Module):
    obs_shape: Tuple
    act_shape: Tuple
    # KWArgs
    net_kwargs: Dict
    critic_kwargs: Dict
    act_kwargs: Dict
    # Hyperparameters
    critic_loss_scale: float
    useless_fine_scale: float
    action_entropy_coef: float
    action_std_scale: float
    ppo_clip: float

    def setup(self):
        self.net = MLPWithGRU(**self.net_kwargs)
        self.critic = MLPHead(**self.critic_kwargs)
        self.action_head = ActorHead(
            **self.act_kwargs, out_features=2 * self.act_shape[-1]
        )

    def ppo_loss_and_grads(self, batch):
        if "hidden" not in batch:
            batch["hidden"] = self.net.initialize_hidden(batch["obs"])
        (
            obs,
            action,
            old_action_log_prob,
            old_action_logstd,
            returns,
            old_value_logits,
            h,
        ) = (
            batch["obs"],
            batch["action"],
            batch["action_logprob"],
            batch["action_logstd"],
            batch["returns"],
            batch["value_logits"],
            batch["hidden"],
        )
        feat, _ = self.net(obs, h)
        feat = np.concatenate([feat, obs], axis=-1)
        action_mean, action_logstd = self.action_head(feat)
        value_logits = self.critic(feat)

        value_dist = C51(value_logits)
        old_value_dist = C51(old_value_logits)

        # Critic loss
        value = value_dist.mean()  # [T0: T0+H+1]
        return_logprob = value_dist.log_prob(sg(returns))
        old_return_logprob = old_value_dist.log_prob(sg(returns))
        clipped_return_logprob = (
            np.minimum(return_logprob - old_return_logprob, np.log(1 + self.ppo_clip))
            + old_return_logprob
        )
        critic_loss = -np.minimum(return_logprob, clipped_return_logprob)

        # Advantages (lot's of messy tricks for stability)
        advantage_raw = returns - sg(value)
        advantage_raw = np.clip(
            advantage_raw, -3, 3
        )  # TODO how to make scale-invariant?
        advantage = (advantage_raw - advantage_raw.mean()) / (
            advantage_raw.std() + 1e-8
        )
        advantage = advantage.squeeze(-1)
        action_logstd = np.clip(
            action_logstd, old_action_logstd - 0.01, old_action_logstd + 0.002
        )
        actor_dist = Normal(action_mean, action_logstd, std_scale=self.action_std_scale)
        action_log_prob_std = actor_dist.log_prob(sg(action))
        action_log_prob = action_log_prob_std
        ratio = np.exp(action_log_prob - old_action_log_prob)
        clipped_ratio = np.clip(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip)
        action_entropy = actor_dist.entropy()
        ppo_loss = -np.minimum(ratio * advantage, clipped_ratio * advantage)
        actor_loss = ppo_loss - self.action_entropy_coef * action_entropy
        useless_move_fine = (nn.relu(np.abs(action_mean) * 3e-3 - 1) ** 2).mean(0)

        # Total loss
        policy_loss = (
            self.critic_loss_scale * critic_loss.mean()
            + actor_loss.mean()
            + self.useless_fine_scale * useless_move_fine.mean()
        )

        # Metrics (some of them are useless, though)
        metric = {
            "ppoLoss": ppo_loss.mean(0),
            "uselessFine": useless_move_fine.mean(-1),
            "actorLoss": actor_loss.mean(0),
            "criticLoss": critic_loss.mean(0).mean(-1),
            "advantage": advantage.mean(0),
            "advantage_raw": advantage_raw.mean(0).mean(-1),
            "critic_error": ((returns - value) ** 2).mean(0).mean(-1),
            "actionEntropy": action_entropy.mean(0),
            "value": value.mean(0).mean(-1),
            "ret": returns.mean(0).mean(-1),
            "rewards": batch["reward"].mean(0).mean(0),
            "logstd_min": action_logstd.min(),
            "logstd_max": action_logstd.max(),
            "logstd_mean": action_logstd.mean(),
        }
        return policy_loss, metric

    def act(self, obs, h):
        feat, h = self.net(obs, h)
        feat = np.concatenate([feat, obs], axis=-1)
        action_mean, action_logstd = self.action_head(feat)
        value_logits = self.critic(feat)
        value = C51(value_logits).mean()
        action, action_logprob = Normal(
            action_mean, action_logstd, std_scale=self.action_std_scale
        ).sample_and_log_prob(seed=self.make_rng("action"))
        return (
            action,
            action_logprob,
            value,
            h,
            value_logits,
            action_logstd,
        )

    def initial_state(self, obs):
        return self.net.initialize_hidden(obs)

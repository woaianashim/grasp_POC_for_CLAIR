import flax.linen as nn
from jax.lax import stop_gradient as sg
import jax.numpy as np
import optax as tx
from src.nets import MLPWithGRU, TQCCritic, ActorHead
from typing import Dict, Tuple
from src.dists import TQD, SquashedNormal


class TQCGraspModel(nn.Module):
    obs_shape: Tuple
    act_shape: Tuple
    # KWArgs
    net_kwargs: Dict
    critic_kwargs: Dict
    act_kwargs: Dict
    # Hyperparameters
    alpha_loss_scale: float
    target_entropy_scale: float
    critic_loss_scale: float
    slow_update_rate: float
    useless_fine_scale: float
    action_std_scale: float
    n_atoms: int
    n_approximations: int
    drop_target_quantiles: int
    gamma: float

    def setup(self):
        self.net = MLPWithGRU(**self.net_kwargs)
        self.critic_kwargs["out_features"] = self.n_atoms
        self.critic = TQCCritic(
            n_critics=self.n_approximations, critic_kwargs=self.critic_kwargs
        )
        self.slow_critic = TQCCritic(
            n_critics=self.n_approximations, critic_kwargs=self.critic_kwargs
        )
        self.critic_sg = TQCCritic(
            n_critics=self.n_approximations, critic_kwargs=self.critic_kwargs
        )  # Copy of critic to make loss possible to run in a single pass
        self.action_head = ActorHead(
            **self.act_kwargs, out_features=2 * self.act_shape[-1]
        )

    def tqc_loss_and_grads(self, batch):
        if "hidden" not in batch:
            batch["hidden"] = self.net.initialize_hidden(batch["obs"])
        (
            obs,  # [ 0,T+1 ]
            action,  # [ 0,T ]
            reward,
            done,
            h,
        ) = (
            batch["obs"],
            batch["action"],
            batch["reward"],
            batch["done"],
            batch["hidden"],
        )
        feat, _ = self.net(obs, h)
        feat = np.concatenate([feat, obs], axis=-1)  # [ 0,T+1 ]
        # Dists
        action_mean, action_logstd = self.action_head(feat)  # [ 0,T+1 ]
        actor_dist = SquashedNormal(
            action_mean, action_logstd, std_scale=self.action_std_scale
        )
        sampled_action, sampled_action_log_prob = actor_dist.sample_and_log_prob(
            seed=self.make_rng("action")
        )  # For action gradient
        value_atoms = self.critic(feat[:-1], action)
        slow_value_atoms = sg(self.slow_critic(feat, sampled_action)[1:])
        sampled_value_atoms = self.critic_sg(sg(feat), sampled_action)[:-1]

        # Alpha loss (sac entropy autocoeff)
        entropy_div = sg(
            -sampled_action_log_prob - action.shape[-1] * self.target_entropy_scale
        ).mean()
        alpha_loss = self.critic.log_alpha * entropy_div
        alpha = sg(np.exp(self.critic.log_alpha))

        # Critic loss
        slow_quantiles = TQD(slow_value_atoms, n_atoms=self.n_atoms).bot_quantiles(
            self.drop_target_quantiles
        )
        next_action_logprob = sampled_action_log_prob[1:]
        target_quantiles = reward + (1 - done[..., None]) * self.gamma * (
            slow_quantiles - sg(alpha * next_action_logprob[..., None])
        )
        deltas = (
            target_quantiles[..., None, :, None] - value_atoms[..., None, :]
        )  # T x B x A x Trgt x V
        quantile_fractions = (2 * np.arange(0, self.n_atoms) + 1.0) / (
            2.0 * self.n_atoms
        )  # 1/2M - 1-1/2M
        critic_loss = (
            sg(np.abs(quantile_fractions - (deltas < 0)))
            * tx.huber_loss(deltas, delta=1)
        ).sum(-1)
        critic_error = deltas**2

        # Actor loss
        sampled_value_dist = TQD(sampled_value_atoms, n_atoms=self.n_atoms)
        sampled_value = sampled_value_dist.mean()
        actor_loss = alpha * sampled_action_log_prob[:-1] - sampled_value

        # Fine for action beyond limit
        useless_move_fine = (nn.relu(np.abs(sampled_action) - 0.05) ** 2).mean(0).sum(
            -1
        ) + tx.huber_loss(nn.relu(np.abs(sampled_action)) ** 2).mean(0).sum(-1) * 0.05

        # Total loss
        policy_loss = (
            self.alpha_loss_scale * alpha_loss
            + self.critic_loss_scale * critic_loss.mean()
            + actor_loss.mean()
            + self.useless_fine_scale * useless_move_fine.mean()
        )

        batch_errors = (
            critic_error.mean(0).mean(-1).mean(-1).mean(-1)
        )  # For priority sampling only

        # Metrics (some of them are useless, though)
        metric = {
            "alphaLoss": alpha_loss,
            "entropy_div": entropy_div,
            "log_alpha": self.critic.log_alpha.mean(),
            "alpha": alpha.mean(),
            "criticLoss": critic_loss.mean(0).mean(-1).mean(-1).mean(-1),
            "criticError": (
                (target_quantiles[..., None, :].mean(-1) - value_atoms.mean(-1))
            ).mean(),
            "criticErrorSq": (
                (target_quantiles[..., None, :].mean(-1) - value_atoms.mean(-1)) ** 2
            ).mean(),
            "atoms": value_atoms.mean(0).mean(0).mean(0),
            "meanQuantiles": sampled_value_dist.mean().mean(),
            "slowQ": slow_quantiles.mean(),
            "targetQ": target_quantiles.mean(),
            "value": sampled_value.mean(),
            "valueQ": value_atoms.mean(),
            "botQuantiles": sampled_value_dist.bot_quantiles(
                self.drop_target_quantiles
            ).mean(),
            "deltas_abs": np.abs(deltas).mean(),
            "deltas": deltas.mean(),
            "dir": (
                np.abs(quantile_fractions - (deltas < 0)) * ((deltas > 0) * 2 - 1)
            ).mean(),
            "salp": (sampled_action_log_prob).mean(),
            "alpha_salp": (
                np.exp(self.critic.log_alpha) * sampled_action_log_prob
            ).mean(),
            # "actionEntropy": sampled_action_entropy.mean(0), # unavailable for SquashedNormal
            "rewards": reward.flatten(),
            "logstd": action_logstd.mean(0).mean(0),
            "useless": useless_move_fine,
            "actions": np.abs(sampled_action).reshape(-1),
            "batch_errors": batch_errors,
        }
        return policy_loss, metric

    def act(self, obs, h):
        feat, h = self.net(obs, h)
        feat = np.concatenate([feat, obs], axis=-1)
        action_mean, action_logstd = self.action_head(feat)
        action, action_logprob = SquashedNormal(
            action_mean, action_logstd, std_scale=self.action_std_scale
        ).sample_and_log_prob(seed=self.make_rng("action"))
        return (
            action,
            h,
        )

    def initial_state(self, obs):
        return self.net.initialize_hidden(obs)

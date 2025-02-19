import jax
import jax.numpy as np
from jax.tree_util import tree_map
from flax.training import checkpoints
import optax
from flax.training.train_state import TrainState as BaseTrainState
from flax.training.dynamic_scale import DynamicScale
from functools import partial
from src.models import TQCGraspModel
import os


class TrainState(
    BaseTrainState
):  # Was used for stability of bfloat16 training, but it has some other bug
    ds: DynamicScale


class TQCGraspAgent:
    def __init__(self, model_cfg, training_params, env=None, dummy_batch=None):
        self._key = jax.random.PRNGKey(0)
        obs_space, act_space = env.observation_space, env.action_space
        self.obs_shape = obs_space.shape
        self.act_shape = act_space.shape
        self.model = TQCGraspModel(
            **model_cfg, obs_shape=obs_space.shape, act_shape=act_space.shape
        )
        self.params = training_params
        self.init_model(dummy_batch)

    def init_model(self, dummy_data):
        self.dobs = dummy_data["obs"]
        policy_variables = self.model.init(
            self.rngs, batch=dummy_data, method=self.model.tqc_loss_and_grads
        )
        params = policy_variables["params"]
        params["slow_critic"] = tree_map(lambda m: m, params["critic"].copy())
        params["critic_sg"] = tree_map(lambda m: m, params["critic"].copy())
        optimizer = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(self.params.lr),
            optax.add_decayed_weights(self.params.weight_decay),
        )
        self.state = TrainState.create(
            apply_fn=partial(self.model.apply, method=self.model.tqc_loss_and_grads),
            params=params,
            tx=optimizer,
            ds=DynamicScale(),
        )
        self.act = jax.jit(partial(self.model.apply, method=self.model.act))
        self.reset_latent(dummy_data["obs"])

    @staticmethod
    @jax.jit
    def _train_policy(state, rngs, batch):

        def loss_fn(params):
            (loss, metric) = state.apply_fn(
                {"params": params},
                batch,
                rngs=rngs,
            )
            return loss, metric

        grad_fn = state.ds.value_and_grad(loss_fn, has_aux=True)
        ds, is_fin, (_, metric), grads = grad_fn(state.params)
        select_fin = partial(np.where, is_fin)

        new_state = state.apply_gradients(grads=grads)
        new_state = new_state.replace(
            ds=ds,
            opt_state=tree_map(select_fin, new_state.opt_state, state.opt_state),
            params=tree_map(select_fin, new_state.params, state.params),
        )
        return new_state, metric

    def train_policy(self, batch):
        new_state, metric = self._train_policy(self.state, self.rngs, batch)
        params = new_state.params
        params["critic_sg"] = tree_map(lambda n: n, params["critic"].copy())
        params["slow_critic"] = tree_map(
            lambda n, o: o * self.model.slow_update_rate
            + n * (1 - self.model.slow_update_rate),
            params["critic"],
            params["slow_critic"],
        )
        self.state = new_state.replace(params=params)

        errors = metric["batch_errors"]
        del metric["batch_errors"]

        return metric, errors

    def train(self, batch):
        policy_metric, errors = self.train_policy(batch)
        return policy_metric, errors

    def step(self, transition, h=None):
        if h is None and self.latent_state is None:
            self.latent_state = self.model.apply(
                self.state.params,
                (
                    transition.obs[None]
                    if len(transition.obs.shape) == 2
                    else transition.obs
                ),
                rngs=self.rngs,
                method=self.model.initial_state,
            )
        (
            transition.action,
            self.latent_state,
        ) = self.act(
            {"params": self.state.params},
            obs=(
                transition.obs[None]
                if len(transition.obs.shape) == 2
                else transition.obs
            ),
            h=h or self.latent_state,
            rngs=self.rngs,
        )
        transition.action = (
            transition.action[0]
            if transition.action.shape[0] == 1
            else transition.action
        )

    def reset_latent(self, obs=None):
        if obs is None:
            self.latent_state = self.latent_state * 0.0
        else:
            self.latent_state = self.model.apply(
                self.state.params,
                obs,
                rngs=self.rngs,
                method=self.model.initial_state,
            )

    @property
    def rngs(self):
        param_key, prior_key, post_key, action_key, skill_key, self._key = (
            jax.random.split(self._key, 6)
        )
        rngs = {
            "params": param_key,
            "prior": prior_key,
            "post": post_key,
            "action": action_key,
            "skill": skill_key,
        }
        return rngs

    def save(self, step):
        ckpt = {
            "model_state": self.state,
            "latent_state": self.latent_state,
            "rngs": self._key,
            "step": step,
        }

        checkpoints.save_checkpoint(
            ckpt_dir=os.path.abspath("./checkpoint"),
            target=ckpt,
            step=step,
            overwrite=True,
            keep=30,
        )

    def load(self, step=None):
        ckpt = {
            "model_state": self.state,
            "latent_state": self.latent_state,
            "rngs": self._key,
            "step": 0,
        }
        print("loading ckpt")
        ckpt = checkpoints.restore_checkpoint(
            ckpt_dir=os.path.abspath("./checkpoint"), step=step, target=ckpt
        )
        print("loaded step", ckpt["step"])
        self.state = ckpt["model_state"]
        self.latent_state = ckpt["latent_state"]
        self._key = ckpt["rngs"]
        return ckpt["step"]

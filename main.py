import hydra
from omegaconf import OmegaConf
from tqdm import tqdm
from src.env import UR5F2Box
from src.ppo_buffer import GPUPPOBuffer
from src.logger import Logger
from src.agent import GraspAgent
import numpy as _np
import os
import logging

os.environ["JAX_PLATFORMS"] = "cpu,gpu"  # Just to avoid annoying jax whining
logging.getLogger("jax._src.xla_bridge").addFilter(
    logging.Filter("Unable to initialize backend")
)


@hydra.main(config_path="config", config_name="conf", version_base=None)
def train(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    env = UR5F2Box(**cfg.env)
    logger = Logger(**cfg.logger)
    buffer = GPUPPOBuffer(**cfg.buffer, env=env)
    agent = GraspAgent(**cfg.agent, env=env, dummy_batch=buffer.dummy_batch)
    if cfg.load:
        logger.info("Loading model")
        initial_step = agent.load(cfg.load_step)
        agent.reset_latent(buffer.dummy_batch["obs"])
        if not cfg.train:
            initial_step = 0
    else:
        initial_step = 0

    transition = {}
    transition["obs"], _ = env.reset()
    transition["done"] = False
    transition["trunc"] = False
    buffer.reset(agent.latent_state)
    for step in tqdm(range(initial_step, cfg.steps)):
        info = inner_loop(env, agent, transition=transition, buffer=buffer)
        if cfg.verbose:
            for k, v in info.items():
                logger.info(
                    f"{k}: min {v.min():.4f} mean {v.mean():.4f} median {_np.median(v):.4f} max {v.max():.4f}"
                )
        if buffer.full:
            if cfg.train:
                next_value = agent.value(transition)
                buffer.prepare_returns(next_value)
                for batch in buffer:
                    metrics = agent.train(batch)
                    logger.log(step, metrics)
            if transition["trunc"]:
                if not cfg.train:
                    break
                transition["obs"], _ = env.reset()
                logger.log(step, info, prefix="env")
                agent.reset_latent()
            buffer.reset(agent.latent_state)

        if step % cfg.save_period == 0 and cfg.save_period > 0:
            agent.save(step)
    env.save_video()


def inner_loop(env, agent, transition={}, buffer=None):
    (
        transition["action"],
        transition["action_logprob"],
        transition["value"],
        transition["value_logits"],
        transition["action_logstd"],
        transition["next_hidden"],
    ) = agent.step(transition)
    (
        transition["next_obs"],
        transition["reward"],
        transition["done"],
        transition["trunc"],
        info,
    ) = env.step(transition["action"])
    if buffer is not None:
        buffer.add(transition)
    transition["obs"] = transition["next_obs"]
    transition["hidden"] = transition["next_hidden"]
    return info


if __name__ == "__main__":
    train()

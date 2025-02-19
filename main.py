import hydra
from omegaconf import OmegaConf
from tqdm import tqdm
from src.env import UR5F2Box
from src.buffer import (
    GPUPPOBuffer,
    BaseEpisodeBuffer,
    HEREpisodeBuffer,
)
from src.logger import Logger
from src.agents import PPOGraspAgent, TQCGraspAgent
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
    if cfg.algo == "ppo":
        BufferClass = GPUPPOBuffer
        AgentClass = PPOGraspAgent
    elif cfg.algo == "tqc":
        if cfg.her:
            BufferClass = HEREpisodeBuffer
        else:
            BufferClass = BaseEpisodeBuffer
        AgentClass = TQCGraspAgent
    buffer = BufferClass(**cfg.buffer, env=env)
    transition = buffer.transition
    agent = AgentClass(**cfg.agent, env=env, dummy_batch=buffer.dummy_batch)
    if cfg.load:
        logger.info("Loading model")
        initial_step = agent.load(cfg.load_step)
        agent.reset_latent(buffer.dummy_batch["obs"])
        if not cfg.train:
            initial_step = 0
    else:
        initial_step = 0

    transition.obses, _ = env.reset()
    buffer.reset(latent_state=agent.latent_state, obses=transition.obses)
    for step in tqdm(range(initial_step, cfg.steps)):
        agent.step(transition)
        (
            transition.obses,
            transition.reward,
            transition.done,
            transition.trunc,
            info,
        ) = env.step(transition.action)
        logger.info(info)
        if transition.trunc:  # TODO check buffer reset PPO
            logger.log(step, info, prefix="env")
            transition.obses, _ = env.reset()
            buffer.reset_obses(transition.obses)
            transition.hidden = agent.reset_latent()
        buffer.add(transition, info)
        if buffer.ready:
            if cfg.train:
                if cfg.algo == "ppo":
                    next_value = agent.value(transition)
                    buffer.prepare_returns(next_value)
                for indeces, batch in buffer:
                    metrics, errors = agent.train(
                        batch
                    )  # Errors used exclusively for curiosity sampling
                    logger.log(step, metrics)
                    buffer.update(indeces, errors)
            if cfg.algo == "ppo":
                buffer.reset(latent_state=agent.latent_state)

        if step % cfg.save_period == 0 and cfg.save_period > 0:
            agent.save(step)
    env.save_video()  # Do nothing if env.show_viewer=False


if __name__ == "__main__":
    train()

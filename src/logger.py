from tensorboardX import SummaryWriter
import jax.numpy as np
import numpy as _np
import hydra
from logging import getLogger

logger = getLogger("Core")


def log_scalar(writer, metrics, step):
    for key in metrics:
        writer.add_scalar(key, metrics[key], step)


class Logger:
    def __init__(self, log_dir=None, verbose=False):
        if log_dir is None:
            log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_dir = log_dir
        self.verbose = verbose
        self.step = 0

    def log(self, step, metrics, prefix="train"):
        self.step = step
        if self.verbose:
            self.info(metrics)
        for key in metrics:
            if (
                isinstance(metrics[key], (np.ndarray, _np.ndarray))
                and metrics[key].size == 1
            ):
                self.writer.add_scalar(prefix + "/" + key, metrics[key], step)
            elif (
                isinstance(metrics[key], (np.ndarray, _np.ndarray))
                and len(metrics[key].shape) == 1
            ):
                self.writer.add_scalars(
                    prefix + "/" + key,
                    {
                        "mean": np.mean(metrics[key]),
                        "min": np.min(metrics[key]),
                        "max": np.max(metrics[key]),
                        "median": np.median(metrics[key]),
                    },
                    step,
                )
            else:
                logger.warning(f"Unprocessed metric key {key}")

    def info(self, info):
        if self.verbose:
            if isinstance(info, str):
                logger.info(info)
            else:
                for k, v in info.items():
                    if isinstance(
                        v,
                        (
                            _np.ndarray,
                            np.ndarray,
                        ),
                    ):
                        logger.info(
                            f"{k}: min {v.min():.4f} mean {v.mean():.4f} median {_np.median(v):.4f} max {v.max():.4f} std {v.std():.4f}"
                        )

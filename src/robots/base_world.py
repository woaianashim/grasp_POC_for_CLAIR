import genesis as gs
from jax.dlpack import from_dlpack as to_jax
from logging import getLogger

logger = getLogger(__name__)


class BaseWorld:
    def __init__(
        self,
        n_envs,
        sticky=10,
        max_t=3200,
        dt=0.01,
        show_viewer=False,
        with_camera=False,
        trunc_is_done=False,
        vis_conf={},
    ):
        self._n_envs = n_envs
        self.sticky = sticky
        self.dt = dt
        self.max_t = max_t
        self._show_viewer = show_viewer
        self.trunc_is_done = trunc_is_done
        self.vis_conf = vis_conf
        self.with_camera = with_camera

    def init_scene(self):
        self.trunc = self.done = False
        self.init_genesis()
        self.scene = gs.Scene(
            show_viewer=self.show_viewer,
            vis_options=gs.options.VisOptions(**self.vis_conf),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                enable_collision=True,
                enable_self_collision=True,
                constraint_solver=gs.constraint_solver.Newton,
                enable_joint_limit=True,
            ),
        )
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )
        if self.with_camera:
            self.cam = self.scene.add_camera(
                res=(1280, 960),
                pos=(3.5, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=False,
            )
        self._init_scene()
        self.scene.build(
            n_envs=self.n_envs, env_spacing=(1.0, 1.0) if self.n_envs < 30 else (0, 0.0)
        )

    def _init_scene(self):
        pass

    def reset(self):
        self._reset()
        if self.with_camera:
            self.cam.start_recording()
        obs = self.get_obs()
        info = self.get_info()
        return to_jax(obs), info

    def _reset(self):
        raise NotImplementedError

    def step(self, action):
        obs, reward, done, trunc, info = self._step(action)
        if self.with_camera:
            self.cam.render()
        return to_jax(obs), to_jax(reward), done, trunc, info

    def _step(self, action):
        raise NotImplementedError

    @classmethod
    def init_genesis(cls):
        if not gs._initialized:
            logger.info("Initializing Genesis")
            gs.init(backend=gs.cuda, logging_level="warning")
        else:
            logger.info("Genesis already initialized")

    def _get_info(self):
        return {}

    def get_info(self):
        info = self._get_info()
        return info

    def get_done(self):
        self.trunc = (
            self.scene.t - self.resetting_steps + 1
        ) >= self.max_t * self.sticky
        self.done = self.trunc if self.trunc_is_done else False
        return self.done, self.trunc

    @property
    def show_viewer(self):
        return self._show_viewer

    @property
    def n_envs(self):
        return self._n_envs

    def save_video(self, filename=None):
        if self.with_camera:
            filename = filename or "show.mp4"
            self.cam.stop_recording(save_to_filename=filename, fps=60)

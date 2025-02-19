from genesis import torch
from .robots.base_world import BaseWorld
from .robots.UR5E import UR5F2
from .robots.objects import Objects
from gymnasium.spaces import Box, Dict
from copy import deepcopy


class UR5F2Box(BaseWorld):
    def __init__(
        self,
        n_envs=1024,
        sticky=1,
        object_configs=[],
        max_t=3200,
        show_viewer=False,
        ur5e_config=None,
        trunc_is_done=False,
        with_camera=False,
        vis_conf={},
        reward_scales={},
        reward_delta=False,
        time_obs=False,
        resetting_steps=20,
        with_goal=False,
    ):
        super().__init__(
            n_envs,
            sticky=sticky,
            max_t=max_t,
            show_viewer=show_viewer,
            trunc_is_done=trunc_is_done,
            vis_conf=vis_conf,
            with_camera=with_camera,
        )
        self.object_configs = object_configs
        self.ur5e_config = ur5e_config
        self.reward_scales = reward_scales
        self.reward_delta = reward_delta
        self.time_obs = time_obs
        self.with_goal = with_goal
        self.resetting_steps = resetting_steps
        self.init_scene()
        self.robot.configure()

    def _init_scene(self):
        self.state = {}
        self.robot = UR5F2(self.scene, self.ur5e_config)
        self.objects = Objects(
            self.scene,
            self.object_configs,
            n_envs=self.n_envs,
            with_goal=self.with_goal,
        )

    def _reset(self):
        self.scene.reset()
        self.robot.reset()
        self.objects.reset()
        for _ in range(self.resetting_steps):
            self.scene.step()
        self.update_state()
        if self.with_goal:
            self.state["goal"] = self.state["core_point"]
        self.state["original_target_pos"] = self.state["target_pos"]
        self.state["reward"] = torch.zeros(self.n_envs).numpy()
        self.state["tot_reward"] = torch.zeros(self.n_envs).numpy()
        self.state["initial_dist"] = self.state["dist"]
        self.old_state = deepcopy(self.state)

    def _step(self, action):
        self.robot.control(action)
        self.state["action"] = torch.from_dlpack(action)
        self.scene.step()
        self.update_state()
        obs = self.get_obs()
        reward, potentials = self.get_reward(
            self.state,
            self.old_state,
        )
        self.state["reward"] = reward
        reward = reward[..., None]  # FIX
        done, trunc = self.get_done()
        info = self.get_info()
        for key, value in potentials.items():
            info[key] = value.cpu().numpy()
        info["reward"] = reward.squeeze().cpu().numpy()
        self.state["tot_reward"] += reward.squeeze().cpu().numpy()
        info["tot_reward"] = self.state["tot_reward"]
        return obs, reward, done, trunc, info

    def update_state(self):
        robot_state = self.robot.state_dict()
        objects_state = self.objects.state_dict()
        self.old_state = deepcopy(self.state)
        self.state.update(robot_state)
        self.state.update(objects_state)
        self.state["dist"] = torch.minimum(
            torch.norm(self.state["target_pos"] - self.state["core_point"], dim=-1),
            torch.tensor(2.0).cuda(),
        )

    def get_obs(self):
        target_oh = self.objects.target.float()
        obs = {
            "robot_pos": self.state["pos"].flatten(1),  # n_geom x 3 -> n_geom*3
            "robot_quat": self.state["quat"].flatten(1),
            "joints": self.state["joints"].contiguous(),
            "joints_vel": self.state["joints_vel"].contiguous(),
            "target_pos": self.state["target_pos"],
            "target_quat": self.state["target_quat"],
            "target_code": target_oh,
            "core_point": self.state["core_point"],
        }
        if self.time_obs:
            obs["time"] = (
                torch.ones_like(self.state["joints"][..., -1:])
                * self.scene.t
                / self.max_t
            )
        if self.with_goal:
            obs["goal"] = self.state["goal"]
        return obs

    def get_reward(self, state, old_state):
        potentials = self._get_potentials(state)
        if self.reward_delta:
            old_potentials = self._get_potentials(old_state)
        reward = 0
        potentials["overdraft"] = state["overdraft"].clip(max=10.0)
        for key, scale in self.reward_scales.items():
            reward += scale * potentials[key]
            if self.reward_delta:
                reward -= scale * old_potentials.get(key, 0)
        return reward, potentials

    def _get_potentials(self, state):
        potentials = {}

        dist = torch.minimum(
            torch.norm(state["target_pos"] - state["core_point"], dim=-1),
            torch.tensor(2.0).to(state["target_pos"].device),
        )

        def rescale(x):  # 1-sigmoid(log(dist)) rescales [0, inf] to [2, 0]
            return 1 / (x + 0.5)

        obj_height = state["target_pos"][..., 2].clip(
            min=state["target_pos"][..., 2] * 0,
            max=state["goal"][..., 2] if self.with_goal else 0.3,
        )

        grasped_coef = rescale(dist)

        potentials["grasped_pos"] = grasped_coef
        potentials["obj_height"] = obj_height
        potentials["grasped_height"] = grasped_coef * obj_height
        potentials["sq_distance"] = dist**2
        potentials["distance"] = dist
        if self.with_goal:
            dist_to_goal = torch.norm(state["target_pos"] - state["goal"])
            potentials["grasped_dist_to_goal"] = rescale(dist_to_goal) * grasped_coef

        return potentials

    def _get_info(self):
        info = {}
        info["target_height"] = (
            (self.state["target_pos"] - self.state["original_target_pos"])[..., 2]
            .cpu()
            .numpy()
        )
        info["dist_to_target"] = self.state["dist"].cpu().numpy()
        info["overdraft"] = self.state["overdraft"].sum(-1).mean().cpu().numpy()
        return info

    @property
    def action_space(self):
        action_space = Box(low=-1, high=1, shape=(self.n_envs, self.robot.action_dim))
        return action_space

    @property
    def observation_space(self):
        n_geoms = self.robot.robot.n_geoms
        n_joints = self.robot.action_dim
        n_obj = len(self.objects)

        def box_space(shape):
            return Box(low=-100, high=100, shape=(self.n_envs,) + shape)

        spaces = {
            "robot_pos": box_space((3 * n_geoms,)),
            "robot_quat": box_space((4 * n_geoms,)),
            "joints": box_space((n_joints,)),
            "joints_vel": box_space((n_joints,)),
            "target_pos": box_space((3,)),
            "target_quat": box_space((4,)),
            "target_code": box_space((n_obj,)),
            "core_point": box_space((3,)),
        }
        if self.time_obs:
            spaces["time"] = box_space((1,))
        if self.with_goal:
            spaces["goal"] = box_space((3,))
        observation_space = Dict(spaces)
        return observation_space

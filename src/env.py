from genesis import torch
from .robots.base_world import BaseWorld
from .robots.UR5E import UR5F2
from .robots.objects import Objects
from gymnasium.spaces import Box


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
        randomize_obj_pos=False,
        with_camera=False,
        vis_conf={},
        reward_scales={},
        resetting_steps=20,
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
        self.randomize_obj_pos = randomize_obj_pos
        self.ur5e_config = ur5e_config
        self.reward_scales = reward_scales
        self.resetting_steps = resetting_steps
        self.init_scene()
        self.robot.configure()

    def _init_scene(self):
        self.state = {}
        self.robot = UR5F2(self.scene, self.ur5e_config)
        self.objects = Objects(self.scene, self.object_configs, n_envs=self.n_envs)

    def _reset(self):
        self.scene.reset()
        self.robot.reset()
        self.objects.reset()
        for _ in range(self.resetting_steps):
            self.update_state()
            self.scene.step()
        self.update_state()
        self.state["original_target_pos"] = self.state["target_pos"]
        self.state["reward"] = torch.zeros(self.n_envs).numpy()
        self.state["tot_reward"] = torch.zeros(self.n_envs).numpy()
        self.state["initial_dist"] = self.state["dist"]

    def _step(self, action):
        self.robot.control(action)
        self.state["action"] = torch.from_dlpack(action)
        for _ in range(self.sticky):
            self.scene.step()
            self.update_state()
        self.update_state()
        obs = self.get_obs()
        reward, rewards = self.get_reward()
        done, trunc = self.get_done()
        info = self.get_info()
        for key, value in rewards.items():
            info[key] = value.cpu().numpy()
            info[key + "_scaled"] = value.cpu().numpy() * self.reward_scales[key]
        info["reward"] = reward.squeeze().cpu().numpy()
        self.state["tot_reward"] += reward.squeeze().cpu().numpy()
        info["tot_reward"] = self.state["tot_reward"]
        return obs, reward, done, trunc, info

    def update_state(self):
        robot_state = self.robot.state_dict()
        objects_state = self.objects.state_dict()
        self.state.update(robot_state)
        self.state.update(objects_state)
        self.state["dist"] = torch.minimum(
            torch.norm(self.state["target_pos"] - self.state["core_point"], dim=-1),
            torch.tensor(2.0).cuda(),
        )
        self.state["asymmetry"] = (
            self.state["left_finger"] - self.state["right_finger"]
        )[..., 2].abs()

    def get_obs(self):
        robot_pose = torch.cat([self.state["pos"], self.state["quat"]], dim=-1).flatten(
            1
        )
        target_oh = self.objects.target.float()
        obs = torch.cat(
            [
                robot_pose,
                self.state["joints"],
                self.state["joints_vel"],
                self.state["target_pos"],
                self.state["target_quat"],
                target_oh,
            ],
            dim=-1,
        )
        return obs

    def get_reward(self):
        rewards = {}
        target_height = torch.clip(
            (self.state["target_pos"] - self.state["original_target_pos"])[..., 2],
            0,
            2,
        )
        rewards["grasped_pos"] = (
            torch.relu(0.025 - self.state["dist"]) / 0.025  # FIX
        ) ** 2
        rewards["grasped_height"] = target_height * rewards["grasped_pos"]
        rewards["overdraft"] = torch.minimum(
            self.state["overdraft"].sum(-1), torch.tensor(10.0).cuda()
        )
        rewards["asymmetry"] = self.state["asymmetry"]
        rewards["sq_distance"] = self.state["dist"] ** 2
        rewards["distance"] = self.state["dist"]
        rewards["ee_quat_delta"] = torch.norm(self.state["ee_quat_delta"], dim=-1)
        reward = 0
        for key, value in rewards.items():
            reward += self.reward_scales[key] * value
        if self.trunc_is_done and self.done:
            raise NotImplementedError
        self.state["reward"] = reward
        return reward[..., None], rewards

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
        n_geoms_obs = self.robot.robot.n_geoms * 7
        n_joints_obs = self.robot.action_dim * 2
        n_obj_obs = 7
        n_obj = len(self.objects)
        observation_space = Box(
            low=-torch.inf,
            high=torch.inf,
            shape=(
                self.n_envs,
                n_geoms_obs + n_joints_obs + n_obj_obs + n_obj,
            ),
        )
        return observation_space

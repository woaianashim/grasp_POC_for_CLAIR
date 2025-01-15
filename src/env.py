from genesis import torch
import genesis as gs
from .robots.base_world import BaseWorld
from .robots.UR5E import UR5F2
from gymnasium.spaces import Box


class UR5F2Box(BaseWorld):
    def __init__(
        self,
        n_envs=1024,
        sticky=1,
        box_size=0.03,
        max_t=3200,
        show_viewer=False,
        ur5e_config=None,
        trunc_is_done=False,
        randomize_box=False,
        with_camera=False,
        vis_conf={},
        reward_scales={},
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
        self.box_size = box_size
        self.randomize_box = randomize_box
        self.ur5e_config = ur5e_config
        self.reward_scales = reward_scales
        self.init_scene()
        self.robot.configure()

    def _init_scene(self):
        self.state = {}
        box_pos = torch.tensor([0.5, 0.1, self.box_size / 2.0])
        self.box = self.scene.add_entity(
            gs.morphs.Box(pos=box_pos, size=(self.box_size,) * 3),
        )
        self.default_box_pos = torch.tile(box_pos, (self.n_envs, 1)).cuda()
        self.box_pos = self.default_box_pos.clone()
        self.robot = UR5F2(self.scene, self.ur5e_config)

    def _reset(self):
        self.scene.reset()
        self.sample_box_pos()
        self.box.set_pos(self.box_pos)
        self.robot.reset()
        self.scene.step()
        self.update_state()
        self.state["reward"] = torch.zeros(self.n_envs).numpy()
        self.state["tot_reward"] = torch.zeros(self.n_envs).numpy()
        self.state["initial_dist"] = self.state["dist"]

    def _step(self, action):
        self.robot.control(action)
        self.state["action"] = torch.from_dlpack(action)
        for _ in range(self.sticky):
            self.scene.step()
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
        box_pos = self.box.get_pos()
        box_quat = self.box.get_quat()
        self.state.update(robot_state)
        self.state["box_pos"] = box_pos
        self.state["box_quat"] = box_quat
        self.state["dist"] = torch.norm(
            self.state["box_pos"] - self.state["core_point"], dim=-1
        )
        self.state["asymmetry"] = (
            self.state["left_finger"] - self.state["right_finger"]
        )[..., 2].abs()

    def get_obs(self):
        robot_pose = torch.cat([self.state["pos"], self.state["quat"]], dim=-1).flatten(
            1
        )
        obs = torch.cat(
            [
                robot_pose,
                self.state["joints"],
                self.state["joints_vel"],
                self.state["box_pos"],
                self.state["box_quat"],
            ],
            dim=-1,
        )
        return obs

    def get_reward(self):
        rewards = {}
        box_height = torch.clip(
            (self.state["box_pos"] / (self.box_size / 2) - 1)[..., 2], 0, 10
        )
        rewards["grasped_pos"] = (
            torch.relu(self.box_size - self.state["dist"]) / self.box_size
        ) ** 2
        rewards["grasped_height"] = box_height * rewards["grasped_pos"]
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
        info["box_height"] = (
            (self.state["box_pos"][..., 2] - self.box_size / 2).cpu().numpy()
        )
        info["dist_to_box"] = self.state["dist"].cpu().numpy()
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
        n_box_obs = 7
        observation_space = Box(
            low=-torch.inf,
            high=torch.inf,
            shape=(
                self.n_envs,
                n_geoms_obs + n_joints_obs + n_box_obs,
            ),
        )
        return observation_space

    def sample_box_pos(self):
        self.box_pos = self.default_box_pos.clone()
        if self.randomize_box:
            ang, rad = torch.rand(
                (
                    2,
                    self.n_envs,
                )
            ).cuda()
            ang *= 2 * torch.pi
            rad = rad * 0.4 + 0.3
            self.box_pos[..., 0] = torch.cos(ang) * rad
            self.box_pos[..., 1] = torch.sin(ang) * rad
            self.box_pos[..., 2] = self.box_size / 2.0
        return self.box_pos

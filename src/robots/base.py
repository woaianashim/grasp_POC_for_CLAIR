import os
import torch
import genesis as gs


class BaseRobot:
    def __init__(self, scene, config):
        self.scene = scene
        self.config = config

    def load_robot(self, path):
        definition_path = os.path.join(os.path.dirname(__file__), "defs", path)
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file=definition_path),
        )

    def configure(self):
        raise NotImplementedError

    def reset(self):
        if self.config.randomize_hand:
            start_pos = (
                torch.rand_like(self.robot.get_qpos())
                * (self.limits[1] - self.limits[0])
                + self.limits[0]
            )
        else:
            start_pos = self.keyframe
        self.robot.set_dofs_position(start_pos)
        self.robot.control_dofs_position(start_pos)
        self._control = start_pos
        self.overdraft = torch.zeros_like(self.robot.get_qpos())
        self._reset()

    def state_dict(self):
        states = {}
        states["pos"] = torch.stack(
            [geom.get_pos() for geom in self.robot.geoms], dim=1
        )
        states["quat"] = torch.stack(
            [geom.get_quat() for geom in self.robot.geoms], dim=1
        )
        states["joints"] = self.robot.get_qpos()
        states["joints_vel"] = self.robot.get_dofs_velocity()
        states["joints_delta"] = states["joints"] - self.keyframe
        states["overdraft"] = self.overdraft
        return states

    @property
    def keyframe(self):
        return torch.from_dlpack(self._keyframe)

    def control(self, action):
        ctype = self.config.control_type
        action = action * self.config.action_scale
        action = torch.from_dlpack(action)
        if ctype == "position":
            action = action * (self.limits[1] - self.limits[0])
            mode = self.config.control_mode
            if mode == "delta":
                self._control = self.robot.get_qpos() + action
            elif mode == "absolute":
                self._control = self.keyframe + action
            elif mode == "direct":
                self._control = action
            else:
                raise NotImplementedError

            self.clip_control()

            self.robot.control_dofs_position(self._control)
        elif ctype == "velocity":
            self._control = 3 * torch.tanh(action)
            self.robot.control_dofs_velocity(self._control)
        else:
            raise NotImplementedError

    def clip_control(self):
        clipped_control = torch.clip(self._control, self.limits[0], self.limits[1])
        self.overdraft = (self._control - clipped_control) ** 2
        self._control = clipped_control

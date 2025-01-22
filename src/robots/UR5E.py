import numpy as _np
import jax.numpy as np
import torch
from .base import BaseRobot


class UR5F2(BaseRobot):
    def __init__(self, scene, config, definition_path="ur5e/UR5E_2_fingers.xml"):
        super().__init__(scene, config)
        self.load_robot(definition_path)

    def configure(self):
        assert self.scene.is_built, "Scene must be built before configuring the robot"
        dofs_idx = self.robot._get_dofs_idx(None)
        self.limits = torch.tensor(
            # fmt: off
            [
                [-2*_np.pi,-1*_np.pi, 0*_np.pi,-1*_np.pi,-1*_np.pi,-1* _np.pi,-0.5,-0.5],
                [ 2*_np.pi, 0*_np.pi, 1*_np.pi, 0*_np.pi, 0*_np.pi, 1* _np.pi, 0.0, 0.0],
            ]
            # fmt: on
        ).cuda()
        self._keyframe = np.array(
            # fmt: off
                [-_np.pi/2, -_np.pi/2, _np.pi/2, -_np.pi/2, -_np.pi/2, 0, -0.25, -0.25]
            # fmt: on
        )
        self.robot._solver.set_dofs_limit(
            self.limits[0], self.limits[1], dofs_idx
        )  # TODO: No native interface, is it working?
        self.robot.set_dofs_kp(
            _np.array([4500.0, 4500.0, 4500.0, 4500.0, 4000.0, 2000.0, 200.0, 200.0])
        )
        self.robot.set_dofs_kv(
            _np.array([450.0, 450.0, 450.0, 450.0, 400.0, 200.0, 20.0, 20.0])
        )

    def state_dict(self):
        states = super().state_dict()
        states["joints"] = states["joints"][..., :-1]
        states["joints_vel"] = states["joints_vel"][..., :-1]
        left = self.robot.get_link("left_inner_finger").get_pos()
        right = self.robot.get_link("right_inner_finger").get_pos()
        ee = self.robot.get_link("ee_link").get_pos()
        ee_quat = self.robot.get_link("ee_link").get_quat()
        states["core_point"] = (left + right) / 2 + (left + right - 2 * ee) * 0.12
        states["left_finger"] = left
        states["right_finger"] = right
        states["ee_quat_delta"] = ee_quat - self.ee_target_quat
        return states

    def control(self, actions):
        actions = np.concat([actions, actions[..., -1:]], axis=-1)
        super().control(actions)

    def _reset(self):
        self.ee_target_quat = self.robot.get_link("ee_link").get_quat()

    @property
    def action_dim(self):
        return self.robot.n_dofs - 1

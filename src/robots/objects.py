import genesis as gs
import numpy as _np
from genesis import torch
from enum import Enum
from typing import Optional, Dict
from dataclasses import dataclass
import os


class ItemKind(Enum):
    box = gs.morphs.Box
    cylinder = gs.morphs.Cylinder
    sphere = gs.morphs.Sphere
    mesh = gs.morphs.Mesh


@dataclass
class Object:
    kind: ItemKind
    params: Dict
    pos: Optional[torch.Tensor] = None
    inst: Optional["gs.Entity"] = None
    n_envs: int = -1

    def sample_pos(self):  # TODO distribution not uniform
        ang, rad = torch.rand(
            (
                2,
                self.n_envs,
            )
        ).cuda()
        ang *= 2 * torch.pi
        rad = rad * 0.4 + 0.3
        z_pos = torch.ones_like(ang) * self.z_pos
        pos = torch.stack([rad * torch.cos(ang), rad * torch.sin(ang), z_pos], dim=-1)
        return pos

    def sample_quat(self):  # TODO same problem
        quat = torch.randn((self.n_envs, 4)).cuda()
        quat = quat / quat.norm(dim=-1, keepdim=True)
        return quat

    @property
    def z_pos(self):
        self._sanitize_params()
        if self.kind == ItemKind.box:
            return (torch.tensor(self.params["size"]) / 2).norm()
        elif self.kind == ItemKind.cylinder:
            return _np.sqrt(
                self.params["radius"] ** 2 + (self.params["height"] / 2) ** 2
            )
        elif self.kind == ItemKind.sphere:
            return self.params["radius"]
        elif self.kind == ItemKind.mesh:
            return 0.2  # TODO
        else:
            raise TypeError

    def instantiate(self, scene):
        self._sanitize_params()
        self.inst = scene.add_entity(self.kind.value(**self.params))

    def _sanitize_params(self):
        if self.kind == ItemKind.box:
            self.params.size = (
                (self.params.size,) * 3
                if isinstance(self.params.size, float)
                else self.params.size
            )
        if self.kind == ItemKind.mesh:
            self.params.file = os.path.join(
                os.path.dirname(__file__), "defs", "objects", self.params.file
            )

    def reset(self):
        self.pos = self.sample_pos()
        self.quat = self.sample_quat()
        self.inst.set_pos(self.pos)
        self.inst.set_quat(self.quat)


class Objects:
    def __init__(self, scene, object_configs, n_envs=-1, with_goal=False):
        self._objects = []
        self.n_envs = n_envs
        self.state = {}
        self.with_goal = with_goal
        for conf in object_configs:
            conf.kind = ItemKind[conf.kind]
            obj = Object(**conf, n_envs=n_envs)
            obj.instantiate(scene)
            obj.sample_pos()
            self._objects.append(obj)

    def __getitem__(self, idx):
        return self._objects[idx].inst

    def reset(self):
        for obj in self._objects:
            obj.reset()
        self.state["target"] = torch.randint(
            0, len(self._objects), (self.n_envs,)
        ).cuda()

    def state_dict(self):
        poses = torch.stack([obj.get_pos() for obj in self], dim=-1)
        quats = torch.stack([obj.get_quat() for obj in self], dim=-1)
        env_ids = torch.arange(0, self.n_envs)
        self.state["target_pos"] = poses[
            env_ids, ..., self.state["target"]
        ].contiguous()
        self.state["target_quat"] = quats[
            env_ids, ..., self.state["target"]
        ].contiguous()
        return self.state

    def __len__(self):
        return len(self._objects)

    @property
    def target(self):
        target = torch.nn.functional.one_hot(self.state["target"], len(self._objects))
        return target

import jax
import jax.numpy as np
import numpy as _np
from enum import StrEnum


class Mode(StrEnum):
    uniform = "uniform"
    curiosity = "curiosity"
    top = "top"


class PrioritiesSampler:
    def __init__(
        self,
        capacity,
        batch_size,
        default_priority=1.0,
        max_priority=1e5,
        beta=0.7,
        alpha=0.7,
        c=1e3,
        eps=0.01,
        seed=42,
        mode=Mode.uniform,
    ):
        self.capacity = capacity
        self.priorities = np.zeros(capacity)
        self.nodes = _np.array([0] * (2 * capacity - 1)).astype(_np.float64)
        self.index_to_count = {}
        self.index_to_visits = {}
        self.count_to_index = {}
        self.size = 0
        self.count = 0
        self.batch_size = batch_size
        self.rng = jax.random.PRNGKey(seed)
        self.max_priority = max_priority if mode != Mode.uniform else 1
        self.alpha = alpha
        self.beta = beta
        self.c = c
        self.eps = eps
        self.mode = mode

    @property
    def total(self):
        return self.nodes[0]

    def reset_count(self):
        self.count = 0

    def update(self, index, loss, skip=False):
        visits = self.index_to_visits[index]
        if skip:
            priority = 0
        elif visits == 0 or self.mode == Mode.uniform:
            priority = self.max_priority
        else:
            priority = self.c * self.beta**visits + (loss + self.eps) ** self.alpha
        count = self.index_to_count[index]
        self.index_to_visits[index] += 1
        idx = count + self.capacity - 1  # child index in tree array

        self.nodes[idx] = priority

        parent = (idx - 1) // 2
        while parent >= 0:
            self.nodes[parent] = self.nodes[2 * parent + 1] + self.nodes[2 * parent + 2]
            parent = (parent - 1) // 2

    def remove(self, index):
        if index in self.index_to_count:
            count = self.index_to_count[index]
            self.update(index, 0, skip=True)
            del self.index_to_visits[index]
            del self.index_to_count[index]
            del self.count_to_index[count]

    def add(self, index, skip=False):
        self.index_to_count[index] = self.count
        self.index_to_visits[index] = 0
        self.count_to_index[self.count] = index
        self.update(index, 0 if skip else self.max_priority, skip=skip)

        self.count = (self.count + 1) % self.capacity

    def sample(self):
        if self.mode == Mode.top:
            scores = self.nodes[self.capacity - 1 :]
            counts = _np.argpartition(-scores, self.batch_size)[: self.batch_size]
        else:
            key, self.rng = jax.random.split(self.rng)
            uniform = jax.random.uniform(
                key, minval=0, maxval=self.total, shape=(self.batch_size,)
            )
            positions = _np.zeros_like(uniform, dtype=np.int32)
            while 2 * positions.min() + 1 < len(self.nodes):
                left, right = 2 * positions + 1, 2 * positions + 2
                left = np.where(left >= len(self.nodes), positions, left)
                right = np.where(right >= len(self.nodes), positions, right)
                left_mask = (uniform <= self.nodes[left]) * (self.nodes[left] > 0) + (
                    self.nodes[right] <= 0
                )
                positions = np.where(left_mask, left, right)
                uniform = np.where(left_mask, uniform, uniform - self.nodes[left])
            counts = positions - self.capacity + 1
        return [self.count_to_index[int(c)] for c in counts]

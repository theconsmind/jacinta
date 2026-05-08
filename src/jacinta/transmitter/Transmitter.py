from __future__ import annotations

import math
import random

from ..utils.scheduler import Scheduler
from .TransmitterNode import TransmitterNode
from .TransmitterSample import TransmitterSample


class Transmitter:
    """ """

    def __init__(
        self,
        min_value: float,
        max_value: float,
        *,
        bias_beta_scale: float = 10.0,
        learning_rate_scheduler: Scheduler,
        hits_scheduler: Scheduler,
        min_interval_width: float | None = None,
        max_depth: int | None = None,
        seed: int | None = None,
    ) -> None:
        """ """
        # min_value & max_value validations
        if not isinstance(min_value, (float, int)):
            raise TypeError("min_value must be a float.")
        if not isinstance(max_value, (float, int)):
            raise TypeError("max_value must be a float.")
        if min_value >= max_value:
            raise ValueError("min_value must be lower than max_value.")
        # bias_beta_scale validations
        if not isinstance(bias_beta_scale, (float, int)):
            raise TypeError("bias_beta_scale must be a float.")
        if bias_beta_scale < 0.0:
            raise ValueError("bias_beta_scale must be greater than or equal to 0.0.")
        # learning_rate_scheduler validations
        if not isinstance(learning_rate_scheduler, Scheduler):
            raise TypeError("learning_rate_scheduler must be a Scheduler.")
        # hits_scheduler validations
        if not isinstance(hits_scheduler, Scheduler):
            raise TypeError("hits_scheduler must be a Scheduler.")
        # min_interval_width validations
        if min_interval_width is not None:
            if not isinstance(min_interval_width, (float, int)):
                raise TypeError("min_interval_width must be a float.")
            if min_interval_width <= 0.0:
                raise ValueError("min_interval_width must be greater than 0.0.")
            if min_interval_width >= max_value - min_value:
                raise ValueError(
                    "min_interval_width must be lower than max_value - min_value."
                )
        # max_depth validations
        if max_depth is not None:
            if not isinstance(max_depth, int):
                raise TypeError("max_depth must be an int.")
            if max_depth < 0:
                raise ValueError("max_depth must be greater than or equal to 0.")
        # seed validations
        if seed is not None:
            if not isinstance(seed, int):
                raise TypeError("seed must be an int.")
        # initializations
        self.bias_beta_scale = float(bias_beta_scale)
        self.learning_rate_scheduler = learning_rate_scheduler
        self.hits_scheduler = hits_scheduler
        self.min_interval_width = (
            float(min_interval_width) if min_interval_width is not None else None
        )
        self.max_depth = max_depth if max_depth is not None else None
        self.rng = random.Random(seed)
        # root node
        root_depth = 0
        root_log_weight = 0.0
        root = TransmitterNode(
            left=float(min_value),
            right=float(max_value),
            parent_id=-1,
            left_child_id=-1,
            right_child_id=-1,
            log_weight=root_log_weight,
            log_mass=math.log(float(max_value) - float(min_value)),
            depth=root_depth,
            hits_left=int(self.hits_scheduler(root_depth)),
        )
        # tree represented as a list of node indices (not pointers)
        self.nodes = [root]
        self.root_id = 0
        return

    def forward(self, bias: float = 0.0) -> TransmitterSample:
        """ """
        # bias validations
        if not isinstance(bias, (float, int)):
            raise TypeError("bias must be a float")
        if not (-1.0 <= bias <= 1.0):
            raise ValueError("bias must be in [-1, 1]")
        # compute bias beta
        bias_beta = 1.0 + float(bias) * self.bias_beta_scale
        # sample a value from the root's log_mass
        node_id = self.root_id
        node = self.nodes[node_id]
        while True:
            # if node is a leaf, sample a uniform value from the leaf's interval
            if node.is_leaf:
                value = self.rng.uniform(node.left, node.right)
                sample = TransmitterSample(value=value, node_id=node_id)
                break
            # bias the search
            left_biased_log_mass = self.nodes[node.left_child_id].log_mass * bias_beta
            right_biased_log_mass = self.nodes[node.right_child_id].log_mass * bias_beta
            # stable log_mass sampling with softmax
            m = max(left_biased_log_mass, right_biased_log_mass)
            left_biased_log_mass = math.exp(left_biased_log_mass - m)
            right_biased_log_mass = math.exp(right_biased_log_mass - m)
            total_biased_log_mass = left_biased_log_mass + right_biased_log_mass
            # decide which child to go to
            r = self.rng.random() * total_biased_log_mass
            if r < left_biased_log_mass:
                node_id = node.left_child_id
            else:
                node_id = node.right_child_id
            node = self.nodes[node_id]
        return sample

    def backward(self, sample: TransmitterSample, feedback: float) -> None:
        """ """
        # sample validations
        if not isinstance(sample, TransmitterSample):
            raise TypeError("sample must be a TransmitterSample")
        # feedback validations
        if not isinstance(feedback, (float, int)):
            raise TypeError("feedback must be a float")
        if not (-1.0 <= feedback <= 1.0):
            raise ValueError("feedback must be in [-1, 1]")
        # find the leaf node that contains the sample
        node_id = sample.node_id
        node = self.nodes[node_id]
        if not node.is_leaf:
            node_id = self._find_leaf(node_id, sample.value)
            node = self.nodes[node_id]
        # update log_weight & log_mass
        node.log_weight += node.learning_rate * float(feedback)
        self._propagate_log_mass_up(node_id)
        # split if needed
        node.hits_left -= 1
        if self._should_split(node_id):
            self._split_leaf(node_id)
        return

    def _find_leaf(self, node_id: int, value: float) -> int:
        """ """
        node = self.nodes[node_id]
        while not node.is_leaf:
            if value < self.nodes[node.left_child_id].right:
                node_id = node.left_child_id
            else:
                node_id = node.right_child_id
            node = self.nodes[node_id]
        return node_id

    def _propagate_log_mass_up(self, node_id: int) -> None:
        """ """
        while node_id != -1:
            node = self.nodes[node_id]
            # update log_mass of leaf node
            if node.is_leaf:
                node.log_mass = node.log_weight + math.log(node.length)
            # update log_mass of internal node
            else:
                left_log_mass = self.nodes[node.left_child_id].log_mass
                right_log_mass = self.nodes[node.right_child_id].log_mass
                m = max(left_log_mass, right_log_mass)
                node.log_mass = m + math.log(
                    math.exp(left_log_mass - m) + math.exp(right_log_mass - m)
                )
            node_id = node.parent_id
        return

    def _should_split(self, node_id: int) -> bool:
        """ """
        node = self.nodes[node_id]
        result = True
        if not node.is_leaf:
            result = False
        elif node.hits_left > 0:
            result = False
        elif (
            self.min_interval_width is not None
            and node.length * 0.5 < self.min_interval_width
        ):
            result = False
        elif self.max_depth is not None and node.depth >= self.max_depth:
            result = False
        return result

    def _split_leaf(self, node_id: int) -> None:
        """ """
        node = self.nodes[node_id]
        if not node.is_leaf:
            raise ValueError("node_id must be a leaf node.")
        node_mid = node.left + (node.right - node.left) * 0.5
        left_child_id = len(self.nodes)
        right_child_id = left_child_id + 1
        child_log_weight = node.log_weight
        child_depth = node.depth + 1
        # create child nodes
        left_child_node = TransmitterNode(
            left=node.left,
            right=node_mid,
            parent_id=node_id,
            left_child_id=-1,
            right_child_id=-1,
            log_weight=child_log_weight,
            hits_left=int(self.hits_scheduler.value(child_depth)),
            log_mass=child_log_weight + math.log(node_mid - node.left),
            depth=child_depth,
            learning_rate=self.learning_rate_scheduler.value(child_depth),
        )
        right_child_node = TransmitterNode(
            left=node_mid,
            right=node.right,
            parent_id=node_id,
            left_child_id=-1,
            right_child_id=-1,
            log_weight=child_log_weight,
            hits_left=int(self.hits_scheduler.value(child_depth)),
            log_mass=child_log_weight + math.log(node.right - node_mid),
            depth=child_depth,
            learning_rate=self.learning_rate_scheduler.value(child_depth),
        )
        self.nodes.append(left_child_node)
        self.nodes.append(right_child_node)
        node.left_child_id = left_child_id
        node.right_child_id = right_child_id
        return

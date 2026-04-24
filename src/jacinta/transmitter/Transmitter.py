from __future__ import annotations

import math
import random

from .TransmitterNode import TransmitterNode
from .TransmitterSample import TransmitterSample
from .TransmitterScheduler import TransmitterScheduler


class Transmitter:
    """
    The Transmitter is a data structure that learns to generate values within a given
    range (min_value, max_value).

    Attributes:
        min_value (float): The minimum value of the range.
        max_value (float): The maximum value of the range.
        hits_scheduler (TransmitterScheduler): The scheduler for the number of hits left
            to split any node.
        learning_rate_scheduler (TransmitterScheduler): The scheduler for the learning
            rate.
        min_weight (float): The minimum weight of a node.
        max_weight (float): The maximum weight of a node.
        min_interval_width (float): The minimum width of an interval.
        max_depth (int | None): The maximum depth of the Transmitter tree.
        seed (int | None): The seed for the random number generator.
        rng (random.Random): The random number generator.
        nodes (list[TransmitterNode]): The list of nodes in the Transmitter tree.
        root_id (int): The ID of the root node.
    """

    def __init__(
        self,
        min_value: float,
        max_value: float,
        *,
        hits_start: int = 1,
        hits_end: int = 1_000_000_000,
        hits_steps: int = 1_000_000_000,
        learning_rate_start: float = 0.01,
        learning_rate_end: float = 0.01,
        learning_rate_steps: int = 1,
        min_weight: float = 1e-9,
        max_weight: float = 1e9,
        min_interval_width: float = 1e-9,
        max_depth: int | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Initialize the Transmitter.

        Args:
            min_value (float): The minimum value of the range.
            max_value (float): The maximum value of the range.
            hits_start (int): The initial number of hits left to split any node.
                Defaults to 1.
            hits_end (int): The final number of hits left to split any node.
                Defaults to 1_000_000_000.
            hits_steps (int): The number of steps for the hits update.
                Defaults to 1_000_000_000.
            learning_rate_start (float): The initial learning rate.
                Defaults to 0.01.
            learning_rate_end (float): The final learning rate.
                Defaults to 0.01.
            learning_rate_steps (int): The number of steps for the learning rate update.
                Defaults to 1.
            min_weight (float): The minimum weight of a node.
                Defaults to 1e-9.
            max_weight (float): The maximum weight of a node.
                Defaults to 1e9.
            min_interval_width (float): The minimum width of an interval.
                Defaults to 1e-9.
            max_depth (int | None): The maximum depth of the Transmitter tree.
                Defaults to None.
            seed (int | None): The seed for the random number generator.
                Defaults to None.
        """
        # min_value & max_value validations
        if not isinstance(min_value, (float, int)):
            raise TypeError("min_value must be a float.")
        if not isinstance(max_value, (float, int)):
            raise TypeError("max_value must be a float.")
        if min_value >= max_value:
            raise ValueError("min_value must be lower than max_value.")
        # hits validations
        if not isinstance(hits_start, int):
            raise TypeError("hits_start must be an int.")
        if hits_start <= 0:
            raise ValueError("hits_start must be greater than 0.")
        if not isinstance(hits_end, int):
            raise TypeError("hits_end must be an int.")
        if hits_end <= 0:
            raise ValueError("hits_end must be greater than 0.")
        if not isinstance(hits_steps, int):
            raise TypeError("hits_steps must be an int.")
        if hits_steps <= 0:
            raise ValueError("hits_steps must be greater than 0.")
        # learning_rate validations
        if not isinstance(learning_rate_start, (float, int)):
            raise TypeError("learning_rate_start must be a float.")
        if learning_rate_start <= 0:
            raise ValueError("learning_rate_start must be greater than 0.")
        if not isinstance(learning_rate_end, (float, int)):
            raise TypeError("learning_rate_end must be a float.")
        if learning_rate_end <= 0:
            raise ValueError("learning_rate_end must be greater than 0.")
        if not isinstance(learning_rate_steps, int):
            raise TypeError("learning_rate_steps must be an int.")
        if learning_rate_steps <= 0:
            raise ValueError("learning_rate_steps must be greater than 0.")
        # min_weight & max_weight validations
        if not isinstance(min_weight, (float, int)):
            raise TypeError("min_weight must be a float.")
        if not isinstance(max_weight, (float, int)):
            raise TypeError("max_weight must be a float.")
        if min_weight <= 0:
            raise ValueError("min_weight must be greater than 0.")
        if min_weight >= max_weight:
            raise ValueError("min_weight must be lower than max_weight.")
        # min_interval_width validations
        if not isinstance(min_interval_width, (float, int)):
            raise TypeError("min_interval_width must be a float.")
        if min_interval_width <= 0:
            raise ValueError("min_interval_width must be greater than 0.")
        # max_depth validations
        if max_depth is not None:
            if not isinstance(max_depth, int):
                raise TypeError("max_depth must be an int.")
            if max_depth <= 0:
                raise ValueError("max_depth must be greater than 0.")
        # seed validations
        if seed is not None:
            if not isinstance(seed, int):
                raise TypeError("seed must be an int.")
        # initializations
        self.hits_scheduler = TransmitterScheduler(hits_start, hits_end, hits_steps)
        self.learning_rate_scheduler = TransmitterScheduler(
            float(learning_rate_start),
            float(learning_rate_end),
            learning_rate_steps,
        )
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.min_interval_width = float(min_interval_width)
        self.max_depth = max_depth
        self.rng = random.Random(seed)
        # tree represented as a list of node indices (not pointers)
        self.nodes: list[TransmitterNode] = []
        # root node
        root_depth = 0
        root_weight = 1.0
        root = TransmitterNode(
            left=float(min_value),
            right=float(max_value),
            parent_id=-1,
            left_child_id=-1,
            right_child_id=-1,
            weight=root_weight,
            hits_left=int(self.hits_scheduler.value(root_depth)),
            mass=root_weight * (float(max_value) - float(min_value)),
            depth=root_depth,
            learning_rate=self.learning_rate_scheduler.value(root_depth),
        )
        self.nodes.append(root)
        self.root_id = 0
        return

    def forward(self, bias: float = 0.0) -> TransmitterSample:
        """
        Generate a sample from the Transmitter.

        Args:
            bias (float): The bias applied to the sampling process.
                Defaults to 0.0.

        Returns:
            TransmitterSample: The sample generated by the Transmitter.
        """
        # bias validations
        if not isinstance(bias, (float, int)):
            raise TypeError("bias must be a float")
        if not (-1.0 <= bias <= 1.0):
            raise ValueError("bias must be in [-1, 1]")
        # normalized bias in ]0, 1[
        bias_eps = 1e-9
        bias = max(min(float(bias), 1.0 - bias_eps), -1.0 + bias_eps)
        bias = 1.0 + math.tan(math.pi * 0.5 * bias)
        # sample a value from the root's mass
        node_id = self.root_id
        node = self.nodes[node_id]
        while True:
            # if node is a leaf, sample a uniform value from the leaf's interval
            if node.is_leaf:
                value = self.rng.uniform(node.left, node.right)
                sample = TransmitterSample(
                    value=value,
                    node_id=node_id,
                )
                break
            # bias the search toward the left or right child
            left_biased_mass = self.nodes[node.left_child_id].mass ** bias
            right_biased_mass = self.nodes[node.right_child_id].mass ** bias
            node_biased_mass = left_biased_mass + right_biased_mass
            # decide which child to go to
            r = self.rng.random() * node_biased_mass
            if r < left_biased_mass:
                node_id = node.left_child_id
            else:
                node_id = node.right_child_id
            node = self.nodes[node_id]
        return sample

    def backward(self, sample: TransmitterSample, feedback: float) -> None:
        """
        Update the Transmitter based on the given feedback.

        Args:
            sample (TransmitterSample): The sample associated with the feedback.
            feedback (float): The feedback from the environment used to update
                the Transmitter.
        """
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
        # update weight & mass
        node.weight = self._clip_weight(
            node.weight * math.exp(node.learning_rate * float(feedback))
        )
        self._propagate_mass_up(node_id)
        # split if needed
        node.hits_left -= 1
        if self._should_split(node_id):
            self._split_leaf(node_id)
        return

    def _find_leaf(self, node_id: int, value: float) -> int:
        """
        Find the leaf node that contains the given value.

        Args:
            node_id (int): The ID of the node to start the search from.
            value (float): The value to find the leaf node for.

        Returns:
            int: The ID of the leaf node that contains the given value.
        """
        node = self.nodes[node_id]
        while not node.is_leaf:
            if value < self.nodes[node.left_child_id].right:
                node_id = node.left_child_id
            else:
                node_id = node.right_child_id
            node = self.nodes[node_id]
        return node_id

    def _clip_weight(self, weight: float) -> float:
        """
        Clip the given weight to the range [min_weight, max_weight].

        Args:
            weight (float): The weight to clip.

        Returns:
            float: The clipped weight.
        """
        if weight < self.min_weight:
            weight = self.min_weight
        elif weight > self.max_weight:
            weight = self.max_weight
        return weight

    def _propagate_mass_up(self, node_id: int) -> None:
        """
        Propagate the mass up the tree.

        Args:
            node_id (int): The ID of the node to start the propagation from.
        """
        while node_id != -1:
            node = self.nodes[node_id]
            # update the mass of the current node
            if node.is_leaf:
                node.mass = node.weight * node.length
            else:
                node.mass = (
                    self.nodes[node.left_child_id].mass
                    + self.nodes[node.right_child_id].mass
                )
            node_id = self.nodes[node_id].parent_id
        return

    def _should_split(self, node_id: int) -> bool:
        """
        Determine if the given node should be split.

        Args:
            node_id (int): The ID of the node to check.

        Returns:
            bool: True if the node should be split, False otherwise.
        """
        node = self.nodes[node_id]
        result = True
        if not node.is_leaf:
            result = False
        elif node.hits_left > 0:
            result = False
        elif self.max_depth is not None and node.depth >= self.max_depth:
            result = False
        elif node.length * 0.5 < self.min_interval_width:
            result = False
        return result

    def _split_leaf(self, node_id: int) -> None:
        """
        Split the given leaf node into two child nodes.

        Args:
            node_id (int): The ID of the node to split.
        """
        node = self.nodes[node_id]
        if not node.is_leaf:
            raise ValueError("node_id must be a leaf node.")
        node_mid = (node.left + node.right) * 0.5
        left_child_id = len(self.nodes)
        right_child_id = left_child_id + 1
        child_weight = node.weight
        child_depth = node.depth + 1
        # create child nodes
        left_child_node = TransmitterNode(
            left=node.left,
            right=node_mid,
            parent_id=node_id,
            left_child_id=-1,
            right_child_id=-1,
            weight=child_weight,
            hits_left=int(self.hits_scheduler.value(child_depth)),
            mass=child_weight * (node_mid - node.left),
            depth=child_depth,
            learning_rate=self.learning_rate_scheduler.value(child_depth),
        )
        right_child_node = TransmitterNode(
            left=node_mid,
            right=node.right,
            parent_id=node_id,
            left_child_id=-1,
            right_child_id=-1,
            weight=child_weight,
            hits_left=int(self.hits_scheduler.value(child_depth)),
            mass=child_weight * (node.right - node_mid),
            depth=child_depth,
            learning_rate=self.learning_rate_scheduler.value(child_depth),
        )
        self.nodes.append(left_child_node)
        self.nodes.append(right_child_node)
        node.left_child_id = left_child_id
        node.right_child_id = right_child_id
        return

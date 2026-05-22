from __future__ import annotations

import json
import math
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..utils.scheduler import Scheduler
from .TransmitterNode import TransmitterNode, TransmitterNodeView
from .TransmitterSample import TransmitterSample


class Transmitter:
    """
    A Transmitter is an adaptive interval-based probability distribution represented
    as a binary tree.

    Attributes:
        min_value (float): The closed left bound of the transmitter's interval.
        max_value (float): The open right bound of the transmitter's interval.
        learning_rate_scheduler (Scheduler): The scheduler for the learning rate.
        hits_scheduler (Scheduler): The scheduler for the number of hits needed
            to split a node.
        bias_scheduler (Scheduler): The scheduler for the bias scale.
        min_interval_width (float | None): The minimum width of an interval.
        max_depth (int | None): The maximum depth of the tree.
        rng (random.Random): The random number generator.
        nodes (tuple[TransmitterNode]): The nodes of the tree.
        root_id (int): The ID of the root node.
    """

    __slots__ = (
        "_min_value",
        "_max_value",
        "_learning_rate_scheduler",
        "_hits_scheduler",
        "_bias_scheduler",
        "_min_interval_width",
        "_max_depth",
        "_rng",
        "_nodes",
        "_root_id",
        "_frozen",
    )

    def __init__(
        self,
        min_value: float,
        max_value: float,
        learning_rate_scheduler: Scheduler,
        hits_scheduler: Scheduler,
        bias_scheduler: Scheduler,
        min_interval_width: float | None = None,
        max_depth: int | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Initialize a Transmitter.

        Args:
            min_value (float): The closed left bound of the transmitter's interval.
            max_value (float): The open right bound of the transmitter's interval.
            learning_rate_scheduler (Scheduler): The scheduler for the learning rate.
            hits_scheduler (Scheduler): The scheduler for the number of hits needed
                to split a node.
            bias_scheduler (Scheduler): The scheduler for the bias scale.
            min_interval_width (float | None): The minimum width of an interval.
                Defaults to None.
            max_depth (int | None): The maximum depth of the tree.
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
        # learning_rate_scheduler validations
        if not isinstance(learning_rate_scheduler, Scheduler):
            raise TypeError("learning_rate_scheduler must be a Scheduler.")
        # hits_scheduler validations
        if not isinstance(hits_scheduler, Scheduler):
            raise TypeError("hits_scheduler must be a Scheduler.")
        # bias_scheduler validations
        if not isinstance(bias_scheduler, Scheduler):
            raise TypeError("bias_scheduler must be a Scheduler.")
        # min_interval_width validations
        if min_interval_width is not None:
            if not isinstance(min_interval_width, (float, int)):
                raise TypeError("min_interval_width must be a float.")
            if min_interval_width <= 0.0:
                raise ValueError("min_interval_width must be greater than 0.0.")
            if min_interval_width > max_value - min_value:
                raise ValueError(
                    "min_interval_width must be lower than or equal to "
                    "max_value - min_value."
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
        super().__setattr__("_frozen", False)
        self._min_value = float(min_value)
        self._max_value = float(max_value)
        self._learning_rate_scheduler = learning_rate_scheduler
        self._hits_scheduler = hits_scheduler
        self._bias_scheduler = bias_scheduler
        self._min_interval_width = (
            float(min_interval_width) if min_interval_width is not None else None
        )
        self._max_depth = max_depth if max_depth is not None else None
        self._rng = random.Random(seed)
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
            hits_left=int(self._hits_scheduler(root_depth)),
        )
        # tree represented as a list of node indices (not pointers)
        self._nodes = [root]
        self._root_id = 0
        super().__setattr__("_frozen", True)
        return

    def __repr__(self) -> str:
        """
        Get the representation of the Transmitter.

        Returns:
            str: The representation of the Transmitter.
        """
        result = (
            f"{self.__class__.__name__}("
            f"min_value={self._min_value!r}, "
            f"max_value={self._max_value!r}, "
            f"learning_rate_scheduler={self._learning_rate_scheduler!r}, "
            f"hits_scheduler={self._hits_scheduler!r}, "
            f"bias_scheduler={self._bias_scheduler!r}, "
            f"min_interval_width={self._min_interval_width!r}, "
            f"max_depth={self._max_depth!r}, "
            f"nodes={self._nodes!r})"
        )
        return result

    @property
    def min_value(self) -> float:
        """
        Get the minimum value of the Transmitter.

        Returns:
            float: The minimum value of the Transmitter.
        """
        return self._min_value

    @property
    def max_value(self) -> float:
        """
        Get the maximum value of the Transmitter.

        Returns:
            float: The maximum value of the Transmitter.
        """
        return self._max_value

    @property
    def learning_rate_scheduler(self) -> Scheduler:
        """
        Get the scheduler for the learning rate.

        Returns:
            Scheduler: The scheduler for the learning rate.
        """
        return self._learning_rate_scheduler

    @property
    def hits_scheduler(self) -> Scheduler:
        """
        Get the scheduler for the number of hits needed to split a node.

        Returns:
            Scheduler: The scheduler for the number of hits needed to split a node.
        """
        return self._hits_scheduler

    @property
    def bias_scheduler(self) -> Scheduler:
        """
        Get the scheduler for the bias scale.

        Returns:
            Scheduler: The scheduler for the bias scale.
        """
        return self._bias_scheduler

    @property
    def min_interval_width(self) -> float | None:
        """
        Get the minimum width of an interval.

        Returns:
            float | None: The minimum width of an interval.
        """
        return self._min_interval_width

    @property
    def max_depth(self) -> int | None:
        """
        Get the maximum depth of the tree.

        Returns:
            int | None: The maximum depth of the tree.
        """
        return self._max_depth

    @property
    def nodes(self) -> tuple[TransmitterNodeView]:
        """
        Get the tree of nodes.

        Returns:
            tuple[TransmitterNodeView]: The tree of nodes.
        """
        return tuple(self._nodes)

    def __eq__(self, other: object) -> bool:
        """
        Check if two Transmitters are equal.

        Args:
            other (object): The object to compare with.

        Returns:
            bool: True if the Transmitters are equal, False otherwise.
        """
        # type validations
        if not isinstance(other, Transmitter):
            return NotImplemented
        # equality check
        result = (
            self._min_value == other._min_value
            and self._max_value == other._max_value
            and self._learning_rate_scheduler == other._learning_rate_scheduler
            and self._hits_scheduler == other._hits_scheduler
            and self._bias_scheduler == other._bias_scheduler
            and self._min_interval_width == other._min_interval_width
            and self._max_depth == other._max_depth
            and self._rng.getstate() == other._rng.getstate()
            and self._nodes == other._nodes
        )
        return result

    def forward(self, bias: float = 0.0) -> TransmitterSample:
        """
        Sample a value from the Transmitter distribution.

        Args:
            bias (float): The bias to apply to the sampling.
                Defaults to 0.0.

        Returns:
            TransmitterSample: The sampled value.
        """
        # bias validations
        if not isinstance(bias, (float, int)):
            raise TypeError("bias must be a float.")
        if not (-1.0 <= bias <= 1.0):
            raise ValueError("bias must be in [-1, 1].")
        # sample a value from the root's log_mass
        node_id = self._root_id
        node = self._nodes[node_id]
        while True:
            # if node is a leaf, sample a uniform value from the leaf's interval
            if node.is_leaf:
                value = self._rng.uniform(node.left, node.right)
                while value == node.right:
                    value = self._rng.uniform(node.left, node.right)
                sample = TransmitterSample(value=value, node_id=node_id)
                break
            # bias the search
            bias_scale = 1.0 + float(bias) * self._bias_scheduler(node.depth)
            left_biased_log_mass = self._nodes[node.left_child_id].log_mass * bias_scale
            right_biased_log_mass = (
                self._nodes[node.right_child_id].log_mass * bias_scale
            )
            # stable log_mass sampling with softmax
            m = max(left_biased_log_mass, right_biased_log_mass)
            left_biased_log_mass = math.exp(left_biased_log_mass - m)
            right_biased_log_mass = math.exp(right_biased_log_mass - m)
            total_biased_log_mass = left_biased_log_mass + right_biased_log_mass
            # decide which child to go to
            r = self._rng.random() * total_biased_log_mass
            if r < left_biased_log_mass:
                node_id = node.left_child_id
            else:
                node_id = node.right_child_id
            node = self._nodes[node_id]
        return sample

    def backward(self, sample: TransmitterSample, feedback: float) -> None:
        """
        Update the Transmitter weights based on feedback.

        Args:
            sample (TransmitterSample): The sample to update the weights for.
            feedback (float): The feedback to use for updating the weights.
        """
        # sample validations
        if not isinstance(sample, TransmitterSample):
            raise TypeError("sample must be a TransmitterSample.")
        if not (0 <= sample.node_id < len(self._nodes)):
            raise IndexError("sample.node_id is out of range.")
        if not (
            self._nodes[sample.node_id].left
            <= sample.value
            < self._nodes[sample.node_id].right
        ):
            raise ValueError("sample.value is out of range.")
        # feedback validations
        if not isinstance(feedback, (float, int)):
            raise TypeError("feedback must be a float.")
        if not (-1.0 <= feedback <= 1.0):
            raise ValueError("feedback must be in [-1, 1].")
        # find the leaf node that contains the sample
        node_id = sample.node_id
        node = self._nodes[node_id]
        if not node.is_leaf:
            node_id = self._find_leaf(node_id, sample.value)
            node = self._nodes[node_id]
        # update log_weight & log_mass
        node.log_weight += self._learning_rate_scheduler(node.depth) * float(feedback)
        self._propagate_log_mass_up(node_id)
        # split if needed
        node.hits_left -= 1
        if self._should_split(node_id):
            self._split_leaf(node_id)
        return

    def copy(self) -> Transmitter:
        """
        Get a copy of the Transmitter.

        Returns:
            Transmitter: A copy of the Transmitter.
        """
        result = deepcopy(self)
        return result

    def to_dict(self) -> dict[str, Any]:
        """
        Get the dictionary representation of the Transmitter.

        Returns:
            dict[str, Any]: The dictionary representation of the Transmitter.
        """
        result = {
            "type": self.__class__.__name__,
            "min_value": self._min_value,
            "max_value": self._max_value,
            "learning_rate_scheduler": self._learning_rate_scheduler.to_dict(),
            "hits_scheduler": self._hits_scheduler.to_dict(),
            "bias_scheduler": self._bias_scheduler.to_dict(),
            "min_interval_width": self._min_interval_width,
            "max_depth": self._max_depth,
            "rng": self._rng.getstate(),
            "nodes": tuple(node.to_dict() for node in self._nodes),
        }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Transmitter:
        """
        Create a Transmitter from a dictionary.

        Args:
            data (dict[str, Any]): The dictionary representation of the Transmitter.

        Returns:
            Transmitter: The Transmitter instance.
        """
        # data validations
        if not isinstance(data, dict):
            raise TypeError("data must be a dict.")
        if "type" not in data:
            raise KeyError("data must contain the key 'type'.")
        if data["type"] != cls.__name__:
            raise ValueError(f"data['type'] must be a {cls.__name__}.")
        if "min_value" not in data:
            raise KeyError("data must contain the key 'min_value'.")
        if "max_value" not in data:
            raise KeyError("data must contain the key 'max_value'.")
        if "learning_rate_scheduler" not in data:
            raise KeyError("data must contain the key 'learning_rate_scheduler'.")
        if "hits_scheduler" not in data:
            raise KeyError("data must contain the key 'hits_scheduler'.")
        if "bias_scheduler" not in data:
            raise KeyError("data must contain the key 'bias_scheduler'.")
        if "min_interval_width" not in data:
            raise KeyError("data must contain the key 'min_interval_width'.")
        if "max_depth" not in data:
            raise KeyError("data must contain the key 'max_depth'.")
        if "rng" not in data:
            raise KeyError("data must contain the key 'rng'.")
        if "nodes" not in data:
            raise KeyError("data must contain the key 'nodes'.")
        # initializations
        result = cls(
            data["min_value"],
            data["max_value"],
            Scheduler.from_dict(data["learning_rate_scheduler"]),
            Scheduler.from_dict(data["hits_scheduler"]),
            Scheduler.from_dict(data["bias_scheduler"]),
            data["min_interval_width"],
            data["max_depth"],
        )
        # overwrite the tree
        object.__setattr__(result, "_frozen", False)
        rng_state = data["rng"]
        rng_state[1] = tuple(rng_state[1])
        result._rng.setstate(tuple(rng_state))
        result._nodes = [
            TransmitterNode.from_dict(node_data) for node_data in data["nodes"]
        ]
        object.__setattr__(result, "_frozen", True)
        return result

    def save(self, path: str | Path, overwrite: bool = False) -> None:
        """
        Save the Transmitter to a json file.

        Args:
            path (str | Path): The path to the file.
            overwrite (bool): Whether to overwrite the file if it exists.
        """
        # path validations
        if not isinstance(path, (str, Path)):
            raise TypeError("path must be a string or a Path.")
        # file validations
        path = Path(path)
        if path.suffix != ".json":
            raise ValueError("path must have a .json extension.")
        if not overwrite and path.exists():
            raise FileExistsError(f"path already exists: {path}.")
        # file creation
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=4)
        return

    @classmethod
    def load(cls, path: str | Path) -> Transmitter:
        """
        Load the Transmitter from a json file.

        Args:
            path (str | Path): The path to the file.

        Returns:
            Transmitter: The Transmitter instance.
        """
        # path validations
        if not isinstance(path, (str, Path)):
            raise TypeError("path must be a string or a Path.")
        # file validations
        path = Path(path)
        if path.suffix != ".json":
            raise ValueError("path must have a .json extension.")
        if not path.exists():
            raise FileNotFoundError(f"path does not exist: {path}.")
        # file loading
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        result = cls.from_dict(data)
        return result

    def _find_leaf(self, node_id: int, value: float) -> int:
        """
        Find the leaf node that contains the given value.

        Args:
            node_id (int): The ID of the node to start the search from.
            value (float): The value to search for.

        Returns:
            int: The ID of the leaf node that contains the given value.
        """
        node = self._nodes[node_id]
        while not node.is_leaf:
            if value < self._nodes[node.left_child_id].right:
                node_id = node.left_child_id
            else:
                node_id = node.right_child_id
            node = self._nodes[node_id]
        return node_id

    def _propagate_log_mass_up(self, node_id: int) -> None:
        """
        Update the log_mass of the given node and its ancestors.

        Args:
            node_id (int): The ID of the node to start the update from.
        """
        while node_id != -1:
            node = self._nodes[node_id]
            # update log_mass of leaf node
            if node.is_leaf:
                node.log_mass = node.log_weight + math.log(node.length)
            # update log_mass of internal node
            else:
                left_log_mass = self._nodes[node.left_child_id].log_mass
                right_log_mass = self._nodes[node.right_child_id].log_mass
                m = max(left_log_mass, right_log_mass)
                node.log_mass = m + math.log(
                    math.exp(left_log_mass - m) + math.exp(right_log_mass - m)
                )
            node_id = node.parent_id
        return

    def _should_split(self, node_id: int) -> bool:
        """
        Decide whether a node should be split.

        Args:
            node_id (int): The ID of the node to check.

        Returns:
            bool: True if the node should be split, False otherwise.
        """
        node = self._nodes[node_id]
        result = True
        if not node.is_leaf:
            result = False
        elif node.hits_left > 0:
            result = False
        elif (
            self._min_interval_width is not None
            and node.length * 0.5 < self._min_interval_width
        ):
            result = False
        elif self._max_depth is not None and node.depth >= self._max_depth:
            result = False
        return result

    def _split_leaf(self, node_id: int) -> None:
        """
        Split the given leaf node.

        Args:
            node_id (int): The ID of the node to split.
        """
        node = self._nodes[node_id]
        if not node.is_leaf:
            raise ValueError("node_id must be a leaf node.")
        node_mid = node.left + (node.right - node.left) * 0.5
        left_child_id = len(self._nodes)
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
            log_mass=child_log_weight + math.log(node_mid - node.left),
            depth=child_depth,
            hits_left=int(self._hits_scheduler(child_depth)),
        )
        right_child_node = TransmitterNode(
            left=node_mid,
            right=node.right,
            parent_id=node_id,
            left_child_id=-1,
            right_child_id=-1,
            log_weight=child_log_weight,
            log_mass=child_log_weight + math.log(node.right - node_mid),
            depth=child_depth,
            hits_left=int(self._hits_scheduler(child_depth)),
        )
        # add new nodes to the tree
        self._nodes.append(left_child_node)
        self._nodes.append(right_child_node)
        node.left_child_id = left_child_id
        node.right_child_id = right_child_id
        return

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Set an attribute of the node.

        Args:
            name (str): The name of the attribute.
            value (Any): The value of the attribute.
        """
        # freeze check
        if getattr(self, "_frozen", False):
            raise AttributeError(f"{self.__class__.__name__} is immutable.")
        # set the attribute
        super().__setattr__(name, value)
        return

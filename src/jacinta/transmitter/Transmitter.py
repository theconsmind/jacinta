from __future__ import annotations

import math
import random
from itertools import product
from typing import Any

from ..evaluator import Evaluator
from ..utils.ndspace import NDSpace
from ..utils.scheduler import Scheduler
from .TransmitterSample import TransmitterSample


class Transmitter(NDSpace):
    """
    A Transmitter represents an NDSpace that manages the information transmitted
    by a Receiver.

    Attributes:
        log_weight (float): The log-weight of the transmitter.
        evaluator (Evaluator): The evaluator associated to the transmitter.
        bias_scale_scheduler (Scheduler): The bias scale scheduler.
        learning_rate_scheduler (Scheduler): The learning rate scheduler.
        hits_rate_scheduler (Scheduler): The hits rate scheduler.
        hits_left (float): The number of hits left to split the transmitter.
        rng (random.Random): The random number generator.
        seed (int | None): The seed for the random number generator.
    """

    __slots__ = (
        "_log_weight",
        "_evaluator",
        "_bias_scale_scheduler",
        "_learning_rate_scheduler",
        "_hits_rate_scheduler",
        "_hits_left",
        "_rng",
        "_seed",
    )

    def __init__(
        self,
        bounds: tuple[tuple[float, float], ...],
        evaluator: Evaluator,
        bias_scale_scheduler: Scheduler,
        learning_rate_scheduler: Scheduler,
        hits_rate_scheduler: Scheduler,
        min_width: float | None = None,
        max_depth: int | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Initialize a Transmitter.

        Args:
            bounds (tuple[tuple[float, float], ...]): The bounds of the transmitter.
            evaluator (Evaluator): The evaluator associated to the transmitter.
            bias_scale_scheduler (Scheduler): The bias scale scheduler.
            learning_rate_scheduler (Scheduler): The learning rate scheduler.
            hits_rate_scheduler (Scheduler): The hits rate scheduler.
            min_width (float | None): The minimum width of each dimension of
                the transmitter. Defaults to None.
            max_depth (int | None): The maximum depth of the transmitter.
                Defaults to None.
            seed (int | None): The seed for the random number generator.
                Defaults to None.
        """
        super().__init__(bounds, min_width, max_depth)
        # evaluator validations
        if not isinstance(evaluator, Evaluator):
            raise TypeError("evaluator must be an Evaluator.")
        # bias_scale_scheduler validations
        if not isinstance(bias_scale_scheduler, Scheduler):
            raise TypeError("bias_scale_scheduler must be a Scheduler.")
        # learning_rate_scheduler validations
        if not isinstance(learning_rate_scheduler, Scheduler):
            raise TypeError("learning_rate_scheduler must be a Scheduler.")
        # hits_rate_scheduler validations
        if not isinstance(hits_rate_scheduler, Scheduler):
            raise TypeError("hits_rate_scheduler must be a Scheduler.")
        # seed validations
        if seed is not None:
            if not isinstance(seed, int):
                raise TypeError("seed must be an int.")
        # initializations
        object.__setattr__(self, "_frozen", False)
        self._log_weight = 0.0
        self._evaluator = evaluator
        self._bias_scale_scheduler = bias_scale_scheduler
        self._learning_rate_scheduler = learning_rate_scheduler
        self._hits_rate_scheduler = hits_rate_scheduler
        self._hits_left = hits_rate_scheduler(self._depth)
        self._rng = random.Random(seed)
        self._seed = seed
        object.__setattr__(self, "_frozen", True)
        return

    def __repr__(self) -> str:
        """
        Get the representation of the transmitter.

        Returns:
            str: The representation of the transmitter.
        """
        result = (
            f"{self.__class__.__name__}"
            f"(bounds={self._bounds!r}, "
            f"log_weight={self._log_weight!r}, "
            f"evaluator={self._evaluator!r}, "
            f"bias_scale_scheduler={self._bias_scale_scheduler!r}, "
            f"learning_rate_scheduler={self._learning_rate_scheduler!r}, "
            f"hits_rate_scheduler={self._hits_rate_scheduler!r}, "
            f"hits_left={self._hits_left!r})"
        )
        return result

    @property
    def split_point(self) -> TransmitterSample | None:
        """
        Get the split point of the transmitter.

        Returns:
            TransmitterSample | None: The split point of the transmitter.
        """
        split_point = super().split_point
        if split_point is not None:
            split_point = TransmitterSample(split_point.coordinates)
        return split_point

    @property
    def log_weight(self) -> float:
        """
        Get the log-weight of the transmitter.

        Returns:
            float: The log-weight of the transmitter.
        """
        return self._log_weight

    @property
    def evaluator(self) -> Evaluator:
        """
        Get the evaluator of the transmitter.

        Returns:
            Evaluator: The evaluator of the transmitter.
        """
        return self._evaluator

    @property
    def bias_scale_scheduler(self) -> Scheduler:
        """
        Get the bias scale scheduler of the transmitter.

        Returns:
            Scheduler: The bias scale scheduler of the transmitter.
        """
        return self._bias_scale_scheduler

    @property
    def learning_rate_scheduler(self) -> Scheduler:
        """
        Get the learning rate scheduler of the transmitter.

        Returns:
            Scheduler: The learning rate scheduler of the transmitter.
        """
        return self._learning_rate_scheduler

    @property
    def hits_rate_scheduler(self) -> Scheduler:
        """
        Get the hits rate scheduler of the transmitter.

        Returns:
            Scheduler: The hits rate scheduler of the transmitter.
        """
        return self._hits_rate_scheduler

    @property
    def hits_left(self) -> float:
        """
        Get the number of hits left in the transmitter.

        Returns:
            float: The number of hits left in the transmitter.
        """
        return self._hits_left

    def __eq__(self, other: object) -> bool:
        """
        Check if two transmitters are equal.

        Args:
            other (object): The object to compare with.

        Returns:
            bool: True if the transmitters are equal, False otherwise.
        """
        # other validations
        if type(self) is not type(other):
            return NotImplemented
        # equality check
        result = (
            super().__eq__(other)
            and self._log_weight == other._log_weight
            and self._evaluator == other._evaluator
            and self._bias_scale_scheduler == other._bias_scale_scheduler
            and self._learning_rate_scheduler == other._learning_rate_scheduler
            and self._hits_rate_scheduler == other._hits_rate_scheduler
            and self._hits_left == other._hits_left
        )
        return result

    def forward(self, bias: float = 0.0) -> TransmitterSample:
        """
        Sample a value from the transmitter distribution.

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
        # sample from the transmitter learned distribution
        transmitter = self
        while not transmitter.is_leaf:
            # separate active and inactive children based on hits_left
            active_children = []
            inactive_children = []
            for child in transmitter._children:
                if child._hits_left <= 0.0:
                    active_children.append(child)
                else:
                    inactive_children.append(child)
            transmitters = list(active_children)
            log_weights = [child._log_weight for child in active_children]
            # if there are inactive children, the parent becomes a possible choice
            if inactive_children:
                max_log_weight = max(child._log_weight for child in inactive_children)
                log_weight = max_log_weight
                log_weight += math.log(
                    sum(
                        math.exp(child._log_weight - max_log_weight)
                        for child in inactive_children
                    )
                )
                transmitters.append(transmitter)
                log_weights.append(log_weight)
            # bias the sampling
            bias_scale = 1.0 + float(bias) * transmitter._bias_scale_scheduler(
                transmitter._depth
            )
            log_weights = [log_weight * bias_scale for log_weight in log_weights]
            # stable log_weights sampling with softmax
            max_log_weight = max(log_weights)
            weights = [
                math.exp(log_weight - max_log_weight) for log_weight in log_weights
            ]
            # choose a transmitter based on log_weights
            transmitter = transmitter._rng.choices(transmitters, weights=weights, k=1)[
                0
            ]
        # sample from the transmitter uniform distribution
        coords = tuple(
            transmitter._rng.uniform(lower, upper)
            for lower, upper in transmitter._bounds
        )
        tsample = TransmitterSample(coords)
        return tsample

    def backward(
        self,
        tsample: TransmitterSample,
        feedback: float,
    ) -> None:
        """
        Update the transmitter distribution based on the feedback.

        Args:
            tsample (TransmitterSample): The transmitter sample.
            feedback (float): The feedback to apply to the distribution.
        """
        # tsample validations
        if not isinstance(tsample, TransmitterSample):
            raise TypeError("tsample must be a TransmitterSample.")
        if tsample.nd != self.nd:
            raise ValueError(f"tsample must be {self.nd}D.")
        if tsample not in self:
            raise ValueError("tsample must be contained in self.")
        # feedback validations
        if not isinstance(feedback, (float, int)):
            raise TypeError("feedback must be a float.")
        if not (-1.0 <= feedback <= 1.0):
            raise ValueError("feedback must be in [-1, 1].")
        # hit the transmitter and split if necessary
        transmitter = self.find_leaf(tsample)
        if transmitter._hits_left > 0.0:
            object.__setattr__(transmitter, "_frozen", False)
            transmitter._hits_left -= 1.0
            if transmitter._hits_left <= 0.0:
                if transmitter.can_split():
                    transmitter.split()
            object.__setattr__(transmitter, "_frozen", True)
            if transmitter._parent is not None:
                transmitter = transmitter._parent
        # propagate the feedback up to the root
        advantage = transmitter._evaluator(float(feedback))
        if advantage is not None:
            while transmitter is not None:
                object.__setattr__(transmitter, "_frozen", False)
                transmitter._log_weight += (
                    transmitter._learning_rate_scheduler(transmitter._depth) * advantage
                )
                object.__setattr__(transmitter, "_frozen", True)
                transmitter = transmitter._parent
        return

    def can_split(self) -> bool:
        """
        Check if the transmitter can be split.

        Returns:
            bool: True if the transmitter can be split, False otherwise.
        """
        # the split point is the midpoint of the transmitter
        coords = tuple((lower + upper) / 2 for lower, upper in self._bounds)
        midpoint = TransmitterSample(coords)
        # check if the transmitter is a leaf
        result = True
        if not self.is_leaf:
            result = False
        # check if the transmitter is at max depth
        elif self._max_depth is not None and self._depth == self._max_depth:
            result = False
        # check if the transmitter can be split by the point
        elif self._min_width is not None:
            for coord, (lower, upper) in zip(
                midpoint.coordinates, self._bounds, strict=True
            ):
                lower_width = coord - lower
                upper_width = upper - coord
                # skip new empty bounds (lower == upper)
                if lower_width != 0 and lower_width < self._min_width:
                    result = False
                    break
                if upper_width != 0 and upper_width < self._min_width:
                    result = False
                    break
        return result

    def split(self) -> tuple[Transmitter, ...]:
        """
        Split the transmitter into smaller transmitters.

        Returns:
            tuple[Transmitter, ...]: The sub-transmitters created by the split.
        """
        # self validations
        if not self.can_split():
            raise ValueError("self cannot be split.")
        # the split point is the midpoint of the transmitter
        coords = tuple((lower + upper) / 2 for lower, upper in self._bounds)
        midpoint = TransmitterSample(coords)
        # split the transmitter
        transmitters = []
        # generate all combinations of upper/lower halves
        for directions in product((False, True), repeat=self.nd):
            new_bounds = list(self._bounds)
            is_valid = True
            # build bounds for each sub-transmitter
            for dim, upper_half in enumerate(directions):
                lower, upper = self._bounds[dim]
                if upper_half:
                    new_bound = (midpoint.coordinates[dim], upper)
                else:
                    new_bound = (lower, midpoint.coordinates[dim])
                # skip if the new bound is empty (lower == upper)
                if new_bound[0] == new_bound[1]:
                    is_valid = False
                    break
                new_bounds[dim] = new_bound
            # create new transmitter if valid (lower < upper)
            if is_valid:
                transmitter = self.__class__(
                    tuple(new_bounds),
                    self._evaluator,
                    self._bias_scale_scheduler,
                    self._learning_rate_scheduler,
                    self._hits_rate_scheduler,
                    self._min_width,
                    self._max_depth,
                )
                object.__setattr__(transmitter, "_frozen", False)
                transmitter._parent = self
                transmitter._root = self._root
                transmitter._depth = self._depth + 1
                transmitter._log_weight = self._log_weight
                transmitter._hits_left = self._hits_rate_scheduler(self._depth + 1)
                object.__setattr__(transmitter, "_frozen", True)
                transmitters.append(transmitter)
        transmitters = tuple(transmitters)
        # update children
        object.__setattr__(self, "_frozen", False)
        self._split_point = midpoint
        self._children = transmitters
        object.__setattr__(self, "_frozen", True)
        self._update_height()
        return transmitters

    def collapse(self) -> None:
        """
        Collapse the transmitter by removing its children.
        """
        raise TypeError("Transmitters cannot be collapsed.")

    def to_dict(self) -> dict[str, Any]:
        """
        Get the dictionary representation of the transmitter.

        Returns:
            dict[str, Any]: The dictionary representation of the transmitter.
        """

        def _to_dict(transmitter: Transmitter) -> dict[str, Any]:
            """
            Recursively convert the tree to a dictionary.

            Args:
                transmitter (Transmitter): The transmitter to convert.

            Returns:
                dict[str, Any]: The dictionary representation of the transmitter.
            """
            result = {
                "type": transmitter.__class__.__name__,
                "bounds": transmitter._bounds,
                "log_weight": transmitter._log_weight,
                "evaluator": transmitter._evaluator.to_dict(),
                "bias_scale_scheduler": transmitter._bias_scale_scheduler.to_dict(),
                "learning_rate_scheduler": (
                    transmitter._learning_rate_scheduler.to_dict()
                ),
                "hits_rate_scheduler": transmitter._hits_rate_scheduler.to_dict(),
                "hits_left": transmitter._hits_left,
                "min_width": transmitter._min_width,
                "max_depth": transmitter._max_depth,
                "seed": transmitter._seed,
                "children": (
                    tuple(_to_dict(child) for child in transmitter._children)
                    if not transmitter.is_leaf
                    else None
                ),
            }
            return result

        result = _to_dict(self)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Transmitter:
        """
        Create a transmitter from a dictionary.

        Args:
            data (dict[str, Any]): The dictionary representation of the transmitter.

        Returns:
            Transmitter: The transmitter.
        """

        def _from_dict(
            data: dict[str, Any], parent: Transmitter | None = None
        ) -> Transmitter:
            """
            Recursively convert a dictionary to a tree.

            Args:
                data (dict[str, Any]): The dictionary representation of the transmitter.
                parent (Transmitter | None): The parent of the transmitter.
                    Defaults to None.

            Returns:
                Transmitter: The transmitter.
            """
            # data validations
            if not isinstance(data, dict):
                raise TypeError("data must be a dict.")
            if "type" not in data:
                raise KeyError("data must contain the key 'type'.")
            if data["type"] != cls.__name__:
                raise ValueError(f"data['type'] must be a {cls.__name__}.")
            if "bounds" not in data:
                raise KeyError("data must contain the key 'bounds'.")
            if "min_width" not in data:
                raise KeyError("data must contain the key 'min_width'.")
            if "max_depth" not in data:
                raise KeyError("data must contain the key 'max_depth'.")
            if "children" not in data:
                raise KeyError("data must contain the key 'children'.")
            if "log_weight" not in data:
                raise KeyError("data must contain the key 'log_weight'.")
            if not isinstance(data["log_weight"], (float, int)):
                raise TypeError("data['log_weight'] must be a float.")
            if "evaluator" not in data:
                raise KeyError("data must contain the key 'evaluator'.")
            if "bias_scale_scheduler" not in data:
                raise KeyError("data must contain the key 'bias_scale_scheduler'.")
            if "learning_rate_scheduler" not in data:
                raise KeyError("data must contain the key 'learning_rate_scheduler'.")
            if "hits_rate_scheduler" not in data:
                raise KeyError("data must contain the key 'hits_rate_scheduler'.")
            if "hits_left" not in data:
                raise KeyError("data must contain the key 'hits_left'.")
            if not isinstance(data["hits_left"], (float, int)):
                raise TypeError("data['hits_left'] must be a float.")
            if "seed" not in data:
                raise KeyError("data must contain the key 'seed'.")
            # parent validations
            if parent is not None:
                if parent._max_depth is not None and parent._depth == parent._max_depth:
                    raise ValueError("parent cannot be split.")
                if parent._min_width != data["min_width"]:
                    raise ValueError(
                        "data['min_width'] must be equal to parent._min_width."
                    )
                if parent._max_depth != data["max_depth"]:
                    raise ValueError(
                        "data['max_depth'] must be equal to parent._max_depth."
                    )
            # initializations
            transmitter = cls(
                data["bounds"],
                Evaluator.from_dict(data["evaluator"]),
                Scheduler.from_dict(data["bias_scale_scheduler"]),
                Scheduler.from_dict(data["learning_rate_scheduler"]),
                Scheduler.from_dict(data["hits_rate_scheduler"]),
                data["min_width"],
                data["max_depth"],
                data["seed"],
            )
            # update parent attributes
            object.__setattr__(transmitter, "_frozen", False)
            if parent is not None:
                transmitter._parent = parent
                transmitter._root = parent._root
                transmitter._depth = parent._depth + 1
            transmitter._log_weight = float(data["log_weight"])
            transmitter._hits_left = float(data["hits_left"])
            object.__setattr__(transmitter, "_frozen", True)
            if transmitter._hits_left > transmitter._hits_rate_scheduler(
                transmitter._depth
            ):
                raise ValueError(
                    "data['hits_left'] is not compatible with the hits_rate_scheduler."
                )
            # update children attributes
            if data["children"] is not None:
                children = tuple(
                    _from_dict(child_data, transmitter)
                    for child_data in data["children"]
                )
                # validate split integrity
                expected_transmitter = cls(
                    data["bounds"],
                    Evaluator.from_dict(data["evaluator"]),
                    Scheduler.from_dict(data["bias_scale_scheduler"]),
                    Scheduler.from_dict(data["learning_rate_scheduler"]),
                    Scheduler.from_dict(data["hits_rate_scheduler"]),
                    data["min_width"],
                    data["max_depth"],
                    data["seed"],
                )
                expected_children = expected_transmitter.split()
                actual_bounds = {child._bounds for child in children}
                expected_bounds = {child._bounds for child in expected_children}
                if (
                    len(children) != len(expected_children)
                    or actual_bounds != expected_bounds
                ):
                    raise ValueError("children are not compatible with split_point.")
                object.__setattr__(transmitter, "_frozen", False)
                transmitter._split_point = expected_transmitter._split_point
                transmitter._children = children
                object.__setattr__(transmitter, "_frozen", True)
                transmitter._update_height()
            return transmitter

        transmitter = _from_dict(data)
        return transmitter

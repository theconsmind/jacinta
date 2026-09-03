from __future__ import annotations

import math
import random
from typing import Any

from ...utils.ndspace import NDSpace
from ...utils.schedulers import Scheduler
from ..evaluators import Evaluator
from .transmitter_sample import TransmitterSample


class Transmitter(NDSpace):
    """
    A Transmitter represents an NDSpace that manages the information transmitted
    by a Processor.

    Attributes:
        log_weight (float): The log-weight of the transmitter.
        evaluator (Evaluator): The evaluator associated to the transmitter.
        bias_scale_scheduler (Scheduler): The bias scale scheduler.
        learning_rate_scheduler (Scheduler): The learning rate scheduler.
        hits_rate_scheduler (Scheduler): The hits rate scheduler.
        hits_left (float): The number of hits left to split the transmitter.
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
        transmitter = (
            f"{self.__class__.__name__}"
            f"(bounds={self._bounds!r}, "
            f"log_weight={self._log_weight!r}, "
            f"evaluator={self._evaluator!r}, "
            f"bias_scale_scheduler={self._bias_scale_scheduler!r}, "
            f"learning_rate_scheduler={self._learning_rate_scheduler!r}, "
            f"hits_rate_scheduler={self._hits_rate_scheduler!r}, "
            f"hits_left={self._hits_left!r})"
        )
        return transmitter

    @property
    def split_point(self) -> TransmitterSample | None:
        """
        Get the split point of the transmitter.

        Returns:
            TransmitterSample | None: The split point of the transmitter.
        """
        return self._split_point

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
        is_equal = (
            super().__eq__(other)
            and self._log_weight == other._log_weight
            and self._evaluator == other._evaluator
            and self._bias_scale_scheduler == other._bias_scale_scheduler
            and self._learning_rate_scheduler == other._learning_rate_scheduler
            and self._hits_rate_scheduler == other._hits_rate_scheduler
            and self._hits_left == other._hits_left
        )
        return is_equal

    def __contains__(self, other: object) -> bool:
        """
        Check if a tsample or transmitter is within the bounds of the transmitter.

        Args:
            other (object): The object to check.

        Returns:
            bool: True if the tsample or transmitter is within the bounds
                of the transmitter, False otherwise.
        """
        # other validations
        if not isinstance(other, (TransmitterSample, Transmitter)):
            raise TypeError("other must be a TransmitterSample or a Transmitter.")
        if other.nd != self.nd:
            raise ValueError(f"other must be {self.nd}D.")
        # check if the tsample is within the bounds
        is_in = super().__contains__(other)
        return is_in

    def find_leaf(self, tsample: TransmitterSample) -> Transmitter:
        """
        Find the leaf that contains the tsample.

        Args:
            tsample (TransmitterSample): The tsample to find the leaf for.

        Returns:
            Transmitter: The leaf that contains the tsample.
        """
        # tsample validations
        if not isinstance(tsample, TransmitterSample):
            raise TypeError("tsample must be a TransmitterSample.")
        if tsample.nd != self.nd:
            raise ValueError(f"tsample must be {self.nd}D.")
        if tsample not in self:
            raise ValueError("tsample must be contained in self.")
        # find the leaf that contains the tsample
        transmitter = super().find_leaf(tsample)
        return transmitter

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
        # sample from the transmitter learned distribution
        transmitter = self
        while not transmitter.is_leaf:
            log_weights = [child._log_weight for child in transmitter._children]
            # bias the sampling
            bias_scale = 1.0 + float(bias) * transmitter._bias_scale_scheduler(
                transmitter._depth
            )
            log_weights = [log_weight * bias_scale for log_weight in log_weights]
            # stable log-weights sampling with softmax
            max_log_weight = max(log_weights)
            weights = [
                math.exp(log_weight - max_log_weight) for log_weight in log_weights
            ]
            # choose a transmitter based on log-weights
            transmitter = transmitter._rng.choices(
                transmitter._children, weights=weights, k=1
            )[0]
        # sample from the transmitter uniform distribution
        coords = tuple(
            lower + transmitter._rng.random() * (upper - lower)
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
        # hit the transmitter
        transmitter = self.find_leaf(tsample)
        should_split = False
        if transmitter._hits_left > 0.0:
            object.__setattr__(transmitter, "_frozen", False)
            transmitter._hits_left -= 1.0
            object.__setattr__(transmitter, "_frozen", True)
        if transmitter._hits_left <= 0.0:
            # the split point is the midpoint of the transmitter
            coords = tuple((lower + upper) / 2 for lower, upper in transmitter._bounds)
            midpoint = TransmitterSample(coords)
            if transmitter.can_split(midpoint):
                should_split = True
        # propagate the feedback up to the root
        advantage = transmitter._evaluator(float(feedback))
        if advantage is not None:
            current = transmitter
            while current is not None:
                object.__setattr__(current, "_frozen", False)
                current._log_weight += (
                    current._learning_rate_scheduler(current._depth) * advantage
                )
                object.__setattr__(current, "_frozen", True)
                current = current._parent
        # split the transmitter if necessary
        if should_split:
            transmitter.split(midpoint)
        return

    def can_split(self, tsample: TransmitterSample) -> bool:
        """
        Check if the transmitter can be split.

        Args:
            tsample (TransmitterSample): The tsample to check if the transmitter
                can be split by.

        Returns:
            bool: True if the transmitter can be split, False otherwise.
        """
        # tsample validations
        if not isinstance(tsample, TransmitterSample):
            raise TypeError("tsample must be a TransmitterSample.")
        if tsample.nd != self.nd:
            raise ValueError(f"tsample must be {self.nd}D.")
        if tsample not in self:
            raise ValueError("tsample must be contained in self.")
        # check if the transmitter is a leaf
        is_splittable = super().can_split(tsample)
        return is_splittable

    def split(self, tsample: TransmitterSample) -> tuple[Transmitter, ...]:
        """
        Split the transmitter into smaller transmitters.

        Args:
            tsample (TransmitterSample): The tsample to split the transmitter by.

        Returns:
            tuple[Transmitter, ...]: The sub-transmitters created by the split.
        """
        # tsample validations
        if not isinstance(tsample, TransmitterSample):
            raise TypeError("tsample must be a TransmitterSample.")
        if tsample.nd != self.nd:
            raise ValueError(f"tsample must be {self.nd}D.")
        if tsample not in self:
            raise ValueError("tsample must be contained in self.")
        # self validations
        if not self.can_split(tsample):
            raise RuntimeError("self cannot be split.")
        # split the transmitter
        transmitters = []
        for bounds in self._get_split_bounds(tsample):
            transmitter = self.__class__(
                bounds,
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
            transmitter._hits_left = self._hits_rate_scheduler(self._depth + 1)
            transmitter._rng = self._rng
            object.__setattr__(transmitter, "_frozen", True)
            transmitters.append(transmitter)
        transmitters = tuple(transmitters)
        # update children
        object.__setattr__(self, "_frozen", False)
        self._split_point = tsample
        self._children = transmitters
        object.__setattr__(self, "_frozen", True)
        self._update_height()
        return transmitters

    def collapse(self) -> None:
        """
        Collapse the transmitter by removing its children.
        """
        raise NotImplementedError("Transmitters cannot be collapsed.")

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
            transmitter_data = {
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
                "rng_state": transmitter._rng.getstate(),
            }
            # add split_point
            if transmitter._split_point is not None:
                transmitter_data["split_point"] = transmitter._split_point.to_dict()
            # add children
            if not transmitter.is_leaf:
                transmitter_data["children"] = tuple(
                    _to_dict(child) for child in transmitter._children
                )
            # add min_width and max_depth
            if transmitter._min_width is not None:
                transmitter_data["min_width"] = transmitter._min_width
            if transmitter._max_depth is not None:
                transmitter_data["max_depth"] = transmitter._max_depth
            # add seed
            if transmitter._seed is not None:
                transmitter_data["seed"] = transmitter._seed
            return transmitter_data

        transmitter = _to_dict(self)
        return transmitter

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
            data: dict[str, Any],
            parent: Transmitter | None = None,
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
            if data.get("children") is not None:
                if not isinstance(data["children"], (tuple, list)):
                    raise TypeError("data['children'] must be a tuple.")
            if (data.get("split_point") is None) != (data.get("children") is None):
                raise ValueError(
                    "data['split_point'] and data['children'] must be both None "
                    "or both not None."
                )
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
            if "rng_state" not in data:
                raise KeyError("data must contain the key 'rng_state'.")
            if not isinstance(data["rng_state"], (tuple, list)):
                raise TypeError("data['rng_state'] must be a tuple.")
            if len(data["rng_state"]) != 3:
                raise ValueError("data['rng_state'] must have 3 elements.")
            if not isinstance(data["rng_state"][0], int):
                raise TypeError("data['rng_state'][0] must be an int.")
            if not isinstance(data["rng_state"][1], (tuple, list)):
                raise TypeError("data['rng_state'][1] must be a tuple.")
            if not all(isinstance(x, int) for x in data["rng_state"][1]):
                raise TypeError("All elements of data['rng_state'][1] must be ints.")
            if data["rng_state"][2] is not None:
                if not isinstance(data["rng_state"][2], (float, int)):
                    raise TypeError("data['rng_state'][2] must be a float.")

            # initializations
            transmitter = cls(
                data["bounds"],
                Evaluator.from_dict(data["evaluator"]),
                Scheduler.from_dict(data["bias_scale_scheduler"]),
                Scheduler.from_dict(data["learning_rate_scheduler"]),
                Scheduler.from_dict(data["hits_rate_scheduler"]),
                data.get("min_width"),
                data.get("max_depth"),
                data.get("seed"),
            )
            rng_state = (
                data["rng_state"][0],
                tuple(x for x in data["rng_state"][1]),
                (
                    float(data["rng_state"][2])
                    if data["rng_state"][2] is not None
                    else None
                ),
            )
            # update parent attributes
            object.__setattr__(transmitter, "_frozen", False)
            if parent is not None:
                if parent._min_width != transmitter._min_width:
                    raise ValueError(
                        "data['min_width'] must be equal to parent.min_width."
                    )
                if parent._max_depth != transmitter._max_depth:
                    raise ValueError(
                        "data['max_depth'] must be equal to parent.max_depth."
                    )
                if transmitter._evaluator != parent._evaluator:
                    raise ValueError(
                        "data['evaluator'] must be equal to parent.evaluator."
                    )
                if transmitter._bias_scale_scheduler != parent._bias_scale_scheduler:
                    raise ValueError(
                        "data['bias_scale_scheduler'] must be equal to "
                        "parent.bias_scale_scheduler."
                    )
                if (
                    transmitter._learning_rate_scheduler
                    != parent._learning_rate_scheduler
                ):
                    raise ValueError(
                        "data['learning_rate_scheduler'] must be equal to "
                        "parent.learning_rate_scheduler."
                    )
                if transmitter._hits_rate_scheduler != parent._hits_rate_scheduler:
                    raise ValueError(
                        "data['hits_rate_scheduler'] must be equal to "
                        "parent.hits_rate_scheduler."
                    )
                if rng_state != parent._rng.getstate():
                    raise ValueError(
                        "data['rng_state'] must be equal to the parent rng state."
                    )
                transmitter._parent = parent
                transmitter._root = parent._root
                transmitter._depth = parent._depth + 1
                transmitter._evaluator = parent._evaluator
                transmitter._bias_scale_scheduler = parent._bias_scale_scheduler
                transmitter._learning_rate_scheduler = parent._learning_rate_scheduler
                transmitter._hits_rate_scheduler = parent._hits_rate_scheduler
                transmitter._rng = parent._rng
            else:
                transmitter._rng.setstate(rng_state)
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
            if data.get("children") is not None:
                if (
                    transmitter._max_depth is not None
                    and transmitter._depth == transmitter._max_depth
                ):
                    raise RuntimeError("transmitter cannot be split.")
                split_point = TransmitterSample.from_dict(data["split_point"])
                children = tuple(
                    _from_dict(child_data, transmitter)
                    for child_data in data["children"]
                )
                # validate split integrity
                expected_bounds = set(transmitter._get_split_bounds(split_point))
                actual_bounds = {child._bounds for child in children}
                if (
                    len(children) != len(expected_bounds)
                    or actual_bounds != expected_bounds
                ):
                    raise ValueError("children are not compatible with split_point.")
                object.__setattr__(transmitter, "_frozen", False)
                transmitter._split_point = split_point
                transmitter._children = children
                object.__setattr__(transmitter, "_frozen", True)
                transmitter._update_height()
            return transmitter

        transmitter = _from_dict(data)
        return transmitter

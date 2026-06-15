from __future__ import annotations

from itertools import product
from typing import Any

from ..transmitter import Transmitter, TransmitterSample
from ..utils.ndspace import NDSpace
from ..utils.scheduler import Scheduler
from .ReceiverSample import ReceiverSample


class Receiver(NDSpace):
    """
    A Receiver represents an NDSpace that manages the information received
    by a Jacinta module.

    Attributes:
        transmitter (Transmitter): The transmitter associated to the receiver.
        hits_rate_scheduler (Scheduler): The hits rate scheduler.
        hits_left (float): The number of hits left to split the receiver.
    """

    __slots__ = (
        "_transmitter",
        "_hits_rate_scheduler",
        "_hits_left",
    )

    def __init__(
        self,
        bounds: tuple[tuple[float, float], ...],
        transmitter: Transmitter,
        hits_rate_scheduler: Scheduler,
        min_width: float | None = None,
        max_depth: int | None = None,
    ) -> None:
        """
        Initialize a Receiver.

        Args:
            bounds (tuple[tuple[float, float], ...]): The bounds of the receiver.
            transmitter (Transmitter): The transmitter associated to the receiver.
            hits_rate_scheduler (Scheduler): The hits rate scheduler.
            min_width (float | None): The minimum width of each dimension of
                the receiver. Defaults to None.
            max_depth (int | None): The maximum depth of the receiver.
                Defaults to None.
        """
        super().__init__(bounds, min_width, max_depth)
        # transmitter validations
        if not isinstance(transmitter, Transmitter):
            raise TypeError("transmitter must be a Transmitter.")
        # hits_rate_scheduler validations
        if not isinstance(hits_rate_scheduler, Scheduler):
            raise TypeError("hits_rate_scheduler must be a Scheduler.")
        # initializations
        object.__setattr__(self, "_frozen", False)
        self._transmitter = transmitter
        self._hits_rate_scheduler = hits_rate_scheduler
        self._hits_left = hits_rate_scheduler(self._depth)
        object.__setattr__(self, "_frozen", True)
        return

    def __repr__(self) -> str:
        """
        Get the representation of the receiver.

        Returns:
            str: The representation of the receiver.
        """
        result = (
            f"{self.__class__.__name__}"
            f"(bounds={self._bounds!r}, "
            f"transmitter={self._transmitter!r}, "
            f"hits_rate_scheduler={self._hits_rate_scheduler!r}, "
            f"hits_left={self._hits_left!r})"
        )
        return result

    @property
    def split_point(self) -> ReceiverSample | None:
        """
        Get the split point of the receiver.

        Returns:
            ReceiverSample | None: The split point of the receiver.
        """
        split_point = super().split_point
        if split_point is not None:
            split_point = ReceiverSample(split_point.coordinates)
        return split_point

    @property
    def transmitter(self) -> Transmitter:
        """
        Get the transmitter of the receiver.

        Returns:
            Transmitter: The transmitter of the receiver.
        """
        return self._transmitter

    @property
    def hits_rate_scheduler(self) -> Scheduler:
        """
        Get the hits rate scheduler of the receiver.

        Returns:
            Scheduler: The hits rate scheduler of the receiver.
        """
        return self._hits_rate_scheduler

    @property
    def hits_left(self) -> float:
        """
        Get the number of hits left in the receiver.

        Returns:
            float: The number of hits left in the receiver.
        """
        return self._hits_left

    def __eq__(self, other: object) -> bool:
        """
        Check if two receivers are equal.

        Args:
            other (object): The object to compare with.

        Returns:
            bool: True if the receivers are equal, False otherwise.
        """
        # other validations
        if type(self) is not type(other):
            return NotImplemented
        # equality check
        result = (
            super().__eq__(other)
            and self._transmitter == other._transmitter
            and self._hits_rate_scheduler == other._hits_rate_scheduler
            and self._hits_left == other._hits_left
        )
        return result

    def forward(self, rsample: ReceiverSample, bias: float = 0.0) -> TransmitterSample:
        """
        Sample a value from the receiver distribution.

        Args:
            rsample (ReceiverSample): The receiver sample.
            bias (float): The bias to apply to the sampling.
                Defaults to 0.0.

        Returns:
            TransmitterSample: The sampled value.
        """
        # rsample validations
        if not isinstance(rsample, ReceiverSample):
            raise TypeError("rsample must be a ReceiverSample.")
        if rsample.nd != self.nd:
            raise ValueError(f"rsample must be {self.nd}D.")
        if rsample not in self:
            raise ValueError("rsample must be contained in self.")
        # bias validations
        if not isinstance(bias, (float, int)):
            raise TypeError("bias must be a float.")
        # generate a tsample in the appropriate active receiver
        receiver = self.find_leaf(rsample)
        if receiver._hits_left > 0.0 and receiver._parent is not None:
            receiver = receiver._parent
        tsample = receiver._transmitter.forward(float(bias))
        return tsample

    def backward(
        self,
        rsample: ReceiverSample,
        tsample: TransmitterSample,
        feedback: float,
    ) -> None:
        """
        Update the receiver distribution based on the feedback.

        Args:
            rsample (ReceiverSample): The receiver sample.
            tsample (TransmitterSample): The transmitter sample.
            feedback (float): The feedback to apply to the distribution.
        """
        # rsample validations
        if not isinstance(rsample, ReceiverSample):
            raise TypeError("rsample must be a ReceiverSample.")
        if rsample.nd != self.nd:
            raise ValueError(f"rsample must be {self.nd}D.")
        if rsample not in self:
            raise ValueError("rsample must be contained in self.")
        # tsample validations
        if not isinstance(tsample, TransmitterSample):
            raise TypeError("tsample must be a TransmitterSample.")
        # feedback validations
        if not isinstance(feedback, (float, int)):
            raise TypeError("feedback must be a float.")
        # hit the receiver and split if necessary
        receiver = self.find_leaf(rsample)
        if receiver._hits_left > 0.0:
            object.__setattr__(receiver, "_frozen", False)
            receiver._hits_left -= 1.0
            if receiver._hits_left <= 0.0:
                if receiver._parent is not None:
                    receiver._transmitter = receiver._parent._transmitter.copy()
                if receiver.can_split():
                    receiver.split()
            object.__setattr__(receiver, "_frozen", True)
            if receiver._parent is not None:
                receiver = receiver._parent
        # propagate the feedback up to the root
        receiver._transmitter.backward(tsample, float(feedback))
        while receiver._parent is not None:
            receiver = receiver._parent
            receiver._transmitter.backward(tsample, float(feedback))
        return

    def can_split(self) -> bool:
        """
        Check if the receiver can be split.

        Returns:
            bool: True if the receiver can be split, False otherwise.
        """
        # the split point is the midpoint of the receiver
        coords = tuple((lower + upper) / 2 for lower, upper in self._bounds)
        midpoint = ReceiverSample(coords)
        # check if the receiver is a leaf
        result = True
        if not self.is_leaf:
            result = False
        # check if the receiver is at max depth
        elif self._max_depth is not None and self._depth == self._max_depth:
            result = False
        # check if the receiver can be split by the point
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

    def split(self) -> tuple[Receiver, ...]:
        """
        Split the receiver into smaller receivers.

        Returns:
            tuple[Receiver, ...]: The sub-receivers created by the split.
        """
        # self validations
        if not self.can_split():
            raise ValueError("self cannot be split.")
        # the split point is the midpoint of the receiver
        coords = tuple((lower + upper) / 2 for lower, upper in self._bounds)
        midpoint = ReceiverSample(coords)
        # split the receiver
        receivers = []
        # generate all combinations of upper/lower halves
        for directions in product((False, True), repeat=self.nd):
            new_bounds = list(self._bounds)
            is_valid = True
            # build bounds for each sub-receiver
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
            # create new receiver if valid (lower < upper)
            if is_valid:
                receiver = self.__class__(
                    tuple(new_bounds),
                    self._transmitter,
                    self._hits_rate_scheduler,
                    self._min_width,
                    self._max_depth,
                )
                object.__setattr__(receiver, "_frozen", False)
                receiver._parent = self
                receiver._root = self._root
                receiver._depth = self._depth + 1
                receiver._hits_left = self._hits_rate_scheduler(self._depth + 1)
                object.__setattr__(receiver, "_frozen", True)
                receivers.append(receiver)
        receivers = tuple(receivers)
        # update children
        object.__setattr__(self, "_frozen", False)
        self._split_point = midpoint
        self._children = receivers
        object.__setattr__(self, "_frozen", True)
        self._update_height()
        return receivers

    def collapse(self) -> None:
        """
        Collapse the receiver by removing its children.
        """
        raise TypeError("Receivers cannot be collapsed.")

    def to_dict(self) -> dict[str, Any]:
        """
        Get the dictionary representation of the receiver.

        Returns:
            dict[str, Any]: The dictionary representation of the receiver.
        """

        def _to_dict(receiver: Receiver) -> dict[str, Any]:
            """
            Recursively convert the tree to a dictionary.

            Args:
                receiver (Receiver): The receiver to convert.

            Returns:
                dict[str, Any]: The dictionary representation of the receiver.
            """
            result = {
                "type": receiver.__class__.__name__,
                "bounds": receiver._bounds,
                "transmitter": receiver._transmitter.to_dict(),
                "hits_rate_scheduler": receiver._hits_rate_scheduler.to_dict(),
                "hits_left": receiver._hits_left,
                "min_width": receiver._min_width,
                "max_depth": receiver._max_depth,
                "children": (
                    tuple(_to_dict(child) for child in receiver._children)
                    if not receiver.is_leaf
                    else None
                ),
            }
            return result

        result = _to_dict(self)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Receiver:
        """
        Create a receiver from a dictionary.

        Args:
            data (dict[str, Any]): The dictionary representation of the receiver.

        Returns:
            Receiver: The receiver.
        """

        def _from_dict(
            data: dict[str, Any], parent: Receiver | None = None
        ) -> Receiver:
            """
            Recursively convert a dictionary to a tree.

            Args:
                data (dict[str, Any]): The dictionary representation of the receiver.
                parent (Receiver | None): The parent of the receiver.
                    Defaults to None.

            Returns:
                Receiver: The receiver.
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
            if "transmitter" not in data:
                raise KeyError("data must contain the key 'transmitter'.")
            if "hits_rate_scheduler" not in data:
                raise KeyError("data must contain the key 'hits_rate_scheduler'.")
            if "hits_left" not in data:
                raise KeyError("data must contain the key 'hits_left'.")
            if not isinstance(data["hits_left"], (float, int)):
                raise TypeError("data['hits_left'] must be a float.")
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
            receiver = cls(
                data["bounds"],
                Transmitter.from_dict(data["transmitter"]),
                Scheduler.from_dict(data["hits_rate_scheduler"]),
                data["min_width"],
                data["max_depth"],
            )
            # update parent attributes
            object.__setattr__(receiver, "_frozen", False)
            if parent is not None:
                receiver._parent = parent
                receiver._root = parent._root
                receiver._depth = parent._depth + 1
            receiver._hits_left = float(data["hits_left"])
            object.__setattr__(receiver, "_frozen", True)
            if receiver._hits_left > receiver._hits_rate_scheduler(receiver._depth):
                raise ValueError(
                    "data['hits_left'] is not compatible with the hits_rate_scheduler."
                )
            # update children attributes
            if data["children"] is not None:
                children = tuple(
                    _from_dict(child_data, receiver) for child_data in data["children"]
                )
                # validate split integrity
                expected_receiver = cls(
                    data["bounds"],
                    Transmitter.from_dict(data["transmitter"]),
                    Scheduler.from_dict(data["hits_rate_scheduler"]),
                    data["min_width"],
                    data["max_depth"],
                )
                expected_children = expected_receiver.split()
                actual_bounds = {child._bounds for child in children}
                expected_bounds = {child._bounds for child in expected_children}
                if (
                    len(children) != len(expected_children)
                    or actual_bounds != expected_bounds
                ):
                    raise ValueError("children are not compatible with split_point.")
                object.__setattr__(receiver, "_frozen", False)
                receiver._split_point = expected_receiver._split_point
                receiver._children = children
                object.__setattr__(receiver, "_frozen", True)
                receiver._update_height()
            return receiver

        receiver = _from_dict(data)
        return receiver

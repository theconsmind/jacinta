from __future__ import annotations

from typing import Any

from ...utils.ndspace import NDSpace
from ...utils.schedulers import Scheduler
from ..transmitter import Transmitter, TransmitterSample
from .receiver_sample import ReceiverSample


class Receiver(NDSpace):
    """
    A Receiver represents an NDSpace that manages the information received
    by a Processor.

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
        receiver = (
            f"{self.__class__.__name__}"
            f"(bounds={self._bounds!r}, "
            f"transmitter={self._transmitter!r}, "
            f"hits_rate_scheduler={self._hits_rate_scheduler!r}, "
            f"hits_left={self._hits_left!r})"
        )
        return receiver

    @property
    def split_point(self) -> ReceiverSample | None:
        """
        Get the split point of the receiver.

        Returns:
            ReceiverSample | None: The split point of the receiver.
        """
        return self._split_point

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
        is_equal = (
            super().__eq__(other)
            and self._transmitter == other._transmitter
            and self._hits_rate_scheduler == other._hits_rate_scheduler
            and self._hits_left == other._hits_left
        )
        return is_equal

    def __contains__(self, other: object) -> bool:
        """
        Check if an rsample or receiver is within the bounds of the receiver.

        Args:
            other (object): The object to check.

        Returns:
            bool: True if the rsample or receiver is within the bounds of the receiver,
                False otherwise.
        """
        # other validations
        if not isinstance(other, (ReceiverSample, Receiver)):
            raise TypeError("other must be a ReceiverSample or a Receiver.")
        if other.nd != self.nd:
            raise ValueError(f"other must be {self.nd}D.")
        # check if the rsample is within the bounds
        is_in = super().__contains__(other)
        return is_in

    def find_leaf(self, rsample: ReceiverSample) -> Receiver:
        """
        Find the leaf that contains the rsample.

        Args:
            rsample (ReceiverSample): The rsample to find the leaf for.

        Returns:
            Receiver: The leaf that contains the rsample.
        """
        # rsample validations
        if not isinstance(rsample, ReceiverSample):
            raise TypeError("rsample must be a ReceiverSample.")
        if rsample.nd != self.nd:
            raise ValueError(f"rsample must be {self.nd}D.")
        if rsample not in self:
            raise ValueError("rsample must be contained in self.")
        # find the leaf that contains the rsample
        receiver = super().find_leaf(rsample)
        return receiver

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
        # generate a tsample in the appropriate active receiver
        receiver = self.find_leaf(rsample)
        tsample = receiver._transmitter.forward(bias)
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
        if tsample.nd != self._transmitter.nd:
            raise ValueError(f"tsample must be {self._transmitter.nd}D.")
        if tsample not in self._transmitter:
            raise ValueError("tsample must be contained in self.transmitter.")
        # feedback validations
        if not isinstance(feedback, (float, int)):
            raise TypeError("feedback must be a float.")
        # hit the receiver
        receiver = self.find_leaf(rsample)
        should_split = False
        if receiver._hits_left > 0.0:
            object.__setattr__(receiver, "_frozen", False)
            receiver._hits_left -= 1.0
            object.__setattr__(receiver, "_frozen", True)
        if receiver._hits_left <= 0.0:
            # the split point is the midpoint of the receiver
            coords = tuple((lower + upper) / 2 for lower, upper in receiver._bounds)
            midpoint = ReceiverSample(coords)
            if receiver.can_split(midpoint):
                should_split = True
        # propagate the feedback up to the root
        current = receiver
        while current is not None:
            current._transmitter.backward(tsample, feedback)
            current = current._parent
        # split the receiver if necessary
        if should_split:
            receiver.split(midpoint)
        return

    def can_split(self, rsample: ReceiverSample) -> bool:
        """
        Check if the receiver can be split.

        Args:
            rsample (ReceiverSample): The rsample to check if the receiver
                can be split by.

        Returns:
            bool: True if the receiver can be split, False otherwise.
        """
        # rsample validations
        if not isinstance(rsample, ReceiverSample):
            raise TypeError("rsample must be a ReceiverSample.")
        if rsample.nd != self.nd:
            raise ValueError(f"rsample must be {self.nd}D.")
        if rsample not in self:
            raise ValueError("rsample must be contained in self.")
        # check if the receiver is a leaf
        is_splittable = super().can_split(rsample)
        return is_splittable

    def split(self, rsample: ReceiverSample) -> tuple[Receiver, ...]:
        """
        Split the receiver into smaller receivers.

        Args:
            rsample (ReceiverSample): The rsample to split the receiver by.

        Returns:
            tuple[Receiver, ...]: The sub-receivers created by the split.
        """
        # rsample validations
        if not isinstance(rsample, ReceiverSample):
            raise TypeError("rsample must be a ReceiverSample.")
        if rsample.nd != self.nd:
            raise ValueError(f"rsample must be {self.nd}D.")
        if rsample not in self:
            raise ValueError("rsample must be contained in self.")
        # self validations
        if not self.can_split(rsample):
            raise RuntimeError("self cannot be split.")
        # split the receiver
        receivers = []
        for bounds in self._get_split_bounds(rsample):
            receiver = self.__class__(
                bounds,
                self._transmitter.copy(),
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
        self._split_point = rsample
        self._children = receivers
        object.__setattr__(self, "_frozen", True)
        self._update_height()
        return receivers

    def collapse(self) -> None:
        """
        Collapse the receiver by removing its children.
        """
        raise NotImplementedError("Receivers cannot be collapsed.")

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
            receiver = {
                "type": receiver.__class__.__name__,
                "bounds": receiver._bounds,
                "transmitter": receiver._transmitter.to_dict(),
                "hits_rate_scheduler": receiver._hits_rate_scheduler.to_dict(),
                "hits_left": receiver._hits_left,
                "min_width": receiver._min_width,
                "max_depth": receiver._max_depth,
                "split_point": (
                    receiver._split_point.to_dict()
                    if receiver._split_point is not None
                    else None
                ),
                "children": (
                    tuple(_to_dict(child) for child in receiver._children)
                    if not receiver.is_leaf
                    else None
                ),
            }
            return receiver

        receiver = _to_dict(self)
        return receiver

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
            data: dict[str, Any],
            parent: Receiver | None = None,
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
            if "split_point" not in data:
                raise KeyError("data must contain the key 'split_point'.")
            if "children" not in data:
                raise KeyError("data must contain the key 'children'.")
            if (data["split_point"] is None) != (data["children"] is None):
                raise ValueError(
                    "data['split_point'] and data['children'] must be both None "
                    "or both not None."
                )
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
                    raise RuntimeError("parent cannot be split.")
                if parent._min_width != data["min_width"]:
                    raise ValueError(
                        "data['min_width'] must be equal to parent.min_width."
                    )
                if parent._max_depth != data["max_depth"]:
                    raise ValueError(
                        "data['max_depth'] must be equal to parent.max_depth."
                    )
            # initializations
            receiver = cls(
                data["bounds"],
                Transmitter.from_dict(data["transmitter"]),
                Scheduler.from_dict(data["hits_rate_scheduler"]),
                data["min_width"],
                data["max_depth"],
            )
            if parent is not None:
                if receiver._hits_rate_scheduler != parent._hits_rate_scheduler:
                    raise ValueError(
                        "data['hits_rate_scheduler'] must be equal to "
                        "parent.hits_rate_scheduler."
                    )
                if receiver._transmitter._bounds != parent._transmitter._bounds:
                    raise ValueError(
                        "data['transmitter']['bounds'] must be equal to "
                        "parent.transmitter.bounds."
                    )
                if receiver._transmitter._min_width != parent._transmitter._min_width:
                    raise ValueError(
                        "data['transmitter']['min_width'] must be equal to "
                        "parent.transmitter.min_width."
                    )
                if receiver._transmitter._max_depth != parent._transmitter._max_depth:
                    raise ValueError(
                        "data['transmitter']['max_depth'] must be equal to "
                        "parent.transmitter.max_depth."
                    )
                if (
                    receiver._transmitter._bias_scale_scheduler
                    != parent._transmitter._bias_scale_scheduler
                ):
                    raise ValueError(
                        "data['transmitter']['bias_scale_scheduler'] must be equal "
                        "to parent.transmitter.bias_scale_scheduler."
                    )
                if (
                    receiver._transmitter._learning_rate_scheduler
                    != parent._transmitter._learning_rate_scheduler
                ):
                    raise ValueError(
                        "data['transmitter']['learning_rate_scheduler'] must be "
                        "equal to parent.transmitter.learning_rate_scheduler."
                    )
                if (
                    receiver._transmitter._hits_rate_scheduler
                    != parent._transmitter._hits_rate_scheduler
                ):
                    raise ValueError(
                        "data['transmitter']['hits_rate_scheduler'] must be equal "
                        "to parent.transmitter.hits_rate_scheduler."
                    )
            # update parent attributes
            object.__setattr__(receiver, "_frozen", False)
            if parent is not None:
                receiver._parent = parent
                receiver._root = parent._root
                receiver._depth = parent._depth + 1
                receiver._hits_rate_scheduler = parent._hits_rate_scheduler
            receiver._hits_left = float(data["hits_left"])
            object.__setattr__(receiver, "_frozen", True)
            if receiver._hits_left > receiver._hits_rate_scheduler(receiver._depth):
                raise ValueError(
                    "data['hits_left'] is not compatible with the hits_rate_scheduler."
                )
            # update children attributes
            if data["children"] is not None:
                split_point = ReceiverSample.from_dict(data["split_point"])
                children = tuple(
                    _from_dict(child_data, receiver) for child_data in data["children"]
                )
                # validate split integrity
                expected_bounds = set(receiver._get_split_bounds(split_point))
                actual_bounds = {child._bounds for child in children}
                if (
                    len(children) != len(expected_bounds)
                    or actual_bounds != expected_bounds
                ):
                    raise ValueError("children are not compatible with split_point.")
                object.__setattr__(receiver, "_frozen", False)
                receiver._split_point = split_point
                receiver._children = children
                object.__setattr__(receiver, "_frozen", True)
                receiver._update_height()
            return receiver

        receiver = _from_dict(data)
        return receiver

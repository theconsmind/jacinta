from __future__ import annotations

from itertools import product
from typing import Any

from ..utils.ndspace import NDSpace
from ..utils.scheduler import Scheduler
from .TransmitterSample import TransmitterSample


class Transmitter(NDSpace):
    """ """

    __slots__ = (
        "_hits_rate_scheduler",
        "_hits_left",
    )

    def __init__(
        self,
        bounds: tuple[tuple[float, float], ...],
        hits_rate_scheduler: Scheduler,
        min_width: float | None = None,
        max_depth: int | None = None,
    ) -> None:
        """ """
        super().__init__(bounds, min_width, max_depth)
        # hits_rate_scheduler validations
        if not isinstance(hits_rate_scheduler, Scheduler):
            raise TypeError("hits_rate_scheduler must be a Scheduler.")
        # initializations
        object.__setattr__(self, "_frozen", False)
        self._hits_rate_scheduler = hits_rate_scheduler
        self._hits_left = hits_rate_scheduler(self._depth)
        object.__setattr__(self, "_frozen", True)
        return

    def __repr__(self) -> str:
        """ """
        result = (
            f"{self.__class__.__name__}"
            f"(bounds={self._bounds!r}, "
            f"hits_rate_scheduler={self._hits_rate_scheduler!r}, "
            f"hits_left={self._hits_left!r})"
        )
        return result

    @property
    def split_point(self) -> TransmitterSample | None:
        """ """
        split_point = super().split_point
        if split_point is not None:
            split_point = TransmitterSample(split_point.coordinates)
        return split_point

    @property
    def hits_rate_scheduler(self) -> Scheduler:
        """ """
        return self._hits_rate_scheduler

    @property
    def hits_left(self) -> float:
        """ """
        return self._hits_left

    def __eq__(self, other: object) -> bool:
        """ """
        # other validations
        if type(self) is not type(other):
            return NotImplemented
        # equality check
        result = (
            super().__eq__(other)
            and self._hits_rate_scheduler == other._hits_rate_scheduler
            and self._hits_left == other._hits_left
        )
        return result

    def forward(self, bias: float = 0.0) -> TransmitterSample:
        """ """
        # bias validations
        if not isinstance(bias, (float, int)):
            raise TypeError("bias must be a float.")
        if not (-1.0 <= bias <= 1.0):
            raise ValueError("bias must be in [-1, 1].")
        #
        # tsample = # TODO
        # return tsample
        return

    def backward(
        self,
        tsample: TransmitterSample,
        feedback: float,
    ) -> None:
        """ """
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
                # if transmitter._parent is not None:
                #     # TODO
                #     transmitter._transmitter = transmitter._parent._transmitter.copy()
                if transmitter.can_split():
                    transmitter.split()
            object.__setattr__(transmitter, "_frozen", True)
            if transmitter._parent is not None:
                transmitter = transmitter._parent
        # propagate the feedback up to the root
        # TODO: transmitter._transmitter.backward(tsample, float(feedback))
        while transmitter._parent is not None:
            transmitter = transmitter._parent
            # TODO: transmitter._transmitter.backward(tsample, float(feedback))
        return

    def can_split(self) -> bool:
        """ """
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
        """ """
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
                    self._hits_rate_scheduler,
                    self._min_width,
                    self._max_depth,
                )
                object.__setattr__(transmitter, "_frozen", False)
                transmitter._parent = self
                transmitter._root = self._root
                transmitter._depth = self._depth + 1
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
        """ """
        raise TypeError("Transmitters cannot be collapsed.")

    def to_dict(self) -> dict[str, Any]:
        """ """

        def _to_dict(transmitter: Transmitter) -> dict[str, Any]:
            """ """
            result = {
                "type": transmitter.__class__.__name__,
                "bounds": transmitter._bounds,
                "hits_rate_scheduler": transmitter._hits_rate_scheduler.to_dict(),
                "hits_left": transmitter._hits_left,
                "min_width": transmitter._min_width,
                "max_depth": transmitter._max_depth,
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
        """ """

        def _from_dict(
            data: dict[str, Any], parent: Transmitter | None = None
        ) -> Transmitter:
            """ """
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
            transmitter = cls(
                data["bounds"],
                Scheduler.from_dict(data["hits_rate_scheduler"]),
                data["min_width"],
                data["max_depth"],
            )
            # update parent attributes
            object.__setattr__(transmitter, "_frozen", False)
            if parent is not None:
                transmitter._parent = parent
                transmitter._root = parent._root
                transmitter._depth = parent._depth + 1
            transmitter._hits_left = float(data["hits_left"])
            object.__setattr__(transmitter, "_frozen", True)
            # update children attributes
            if data["children"] is not None:
                children = tuple(
                    _from_dict(child_data, transmitter)
                    for child_data in data["children"]
                )
                # validate split integrity
                expected_transmitter = cls(
                    data["bounds"],
                    Scheduler.from_dict(data["hits_rate_scheduler"]),
                    data["min_width"],
                    data["max_depth"],
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

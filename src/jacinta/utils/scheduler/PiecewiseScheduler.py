from __future__ import annotations

from bisect import bisect_right
from typing import Any

from .Scheduler import Scheduler


class PiecewiseScheduler(Scheduler):
    """
    A Scheduler that uses different Schedulers for different depth ranges.

    Attributes:
        segments (tuple[tuple[int, Scheduler], ...]): The segments of the scheduler.
    """

    __slots__ = ("_segments", "_depths")

    def __init__(
        self,
        segments: tuple[tuple[int, Scheduler], ...],
    ) -> None:
        """
        Initialize a PiecewiseScheduler.

        Args:
            segments (tuple[tuple[int, Scheduler], ...]): The segments of the scheduler.
        """
        # segments validations
        if not isinstance(segments, (tuple, list)):
            raise TypeError("segments must be a tuple.")
        if len(segments) == 0:
            raise ValueError("segments must not be empty.")
        for segment in segments:
            if not isinstance(segment, (tuple, list)):
                raise TypeError("All segments must be tuples.")
            if len(segment) != 2:
                raise ValueError("All segments must have length 2.")
            if not isinstance(segment[0], int):
                raise TypeError("All first elements of segments must be ints.")
            if not isinstance(segment[1], Scheduler):
                raise TypeError("All second elements of segments must be Schedulers.")
        # segment depths validations
        depths = tuple(segment[0] for segment in segments)
        if depths[0] != 0:
            raise ValueError("The first depth must be 0.")
        for idx in range(len(depths) - 1):
            if depths[idx] >= depths[idx + 1]:
                raise ValueError(
                    "segments must be sorted by depth and depths must be unique."
                )
        # initializations
        object.__setattr__(self, "_frozen", False)
        self._segments = tuple(tuple(segment) for segment in segments)
        self._depths = depths
        object.__setattr__(self, "_frozen", True)
        return

    def __repr__(self) -> str:
        """
        Get the representation of the scheduler.

        Returns:
            str: The representation of the scheduler.
        """
        result = f"{self.__class__.__name__}(segments={self._segments!r})"
        return result

    def __call__(self, depth: int) -> float:
        """
        Get the value assigned to the given depth.

        Args:
            depth (int): The depth.

        Returns:
            float: The value assigned to the given depth.
        """
        # depth validations
        if not isinstance(depth, int):
            raise TypeError("depth must be an int.")
        if depth < 0:
            raise ValueError("depth must be greater than or equal to 0.")
        # get the value based on the depth
        idx = bisect_right(self._depths, depth) - 1
        _, scheduler = self._segments[idx]
        result = scheduler(depth)
        return result

    @property
    def segments(self) -> tuple[tuple[int, Scheduler], ...]:
        """
        Get the segments of the scheduler.

        Returns:
            tuple[tuple[int, Scheduler], ...]: The segments of the scheduler.
        """
        return self._segments

    def __eq__(self, other: object) -> bool:
        """
        Check if two schedulers are equal.

        Args:
            other (object): The object to compare with.

        Returns:
            bool: True if the schedulers are equal, False otherwise.
        """
        # other validations
        if type(self) is not type(other):
            return NotImplemented
        # equality check
        result = self._segments == other._segments
        return result

    def to_dict(self) -> dict[str, Any]:
        """
        Get the dictionary representation of the scheduler.

        Returns:
            dict[str, Any]: The dictionary representation of the scheduler.
        """
        result = {
            "type": self.__class__.__name__,
            "segments": tuple(
                (depth, scheduler.to_dict()) for depth, scheduler in self._segments
            ),
        }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PiecewiseScheduler:
        """
        Create a scheduler from a dictionary.

        Args:
            data (dict[str, Any]): The dictionary representation of the scheduler.

        Returns:
            PiecewiseScheduler: The scheduler.
        """
        # data validations
        if not isinstance(data, dict):
            raise TypeError("data must be a dict.")
        if "type" not in data:
            raise KeyError("data must contain the key 'type'.")
        if data["type"] != cls.__name__:
            raise ValueError(f"data['type'] must be a {cls.__name__}.")
        if "segments" not in data:
            raise KeyError("data must contain the key 'segments'.")
        # initializations
        segments = tuple(
            (depth, Scheduler.from_dict(scheduler_data))
            for depth, scheduler_data in data["segments"]
        )
        result = cls(segments)
        return result

from __future__ import annotations

from bisect import bisect_right
from typing import Any

from .ScheduleStrategy import ScheduleStrategy


class PiecewiseScheduleStrategy(ScheduleStrategy):
    """
    A ScheduleStrategy that uses different strategies for different ranges
    of node depths.

    Attributes:
        segments (tuple[tuple[int, ScheduleStrategy]]): The segments of the strategy.
    """

    __slots__ = ("_segments", "_depths")

    def __init__(
        self,
        segments: tuple[tuple[int, ScheduleStrategy]],
    ) -> None:
        """
        Initialize a PiecewiseScheduleStrategy.

        Args:
            segments (tuple[tuple[int, ScheduleStrategy]]): The segments
                of the strategy.
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
            if not isinstance(segment[1], ScheduleStrategy):
                raise TypeError(
                    "All second elements of segments must be ScheduleStrategies."
                )
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
        super().__setattr__("_frozen", False)
        self._segments = tuple(tuple(segment) for segment in segments)
        self._depths = depths
        super().__setattr__("_frozen", True)
        return

    def __repr__(self) -> str:
        """
        Get the representation of the strategy.

        Returns:
            str: The representation of the strategy.
        """
        result = f"{self.__class__.__name__}(segments={self._segments!r})"
        return result

    def __call__(self, depth: int) -> float:
        """
        Get the strategy value based on the node depth.

        Args:
            depth (int): The depth of the node.

        Returns:
            float: The strategy value based on the node depth.
        """
        # depth validations
        if not isinstance(depth, int):
            raise TypeError("depth must be an int.")
        if depth < 0:
            raise ValueError("depth must be greater than or equal to 0.")
        # get the value based on the depth
        idx = bisect_right(self._depths, depth) - 1
        _, strategy = self._segments[idx]
        result = strategy(depth)
        return result

    @property
    def segments(self) -> tuple[tuple[int, ScheduleStrategy]]:
        """
        Get the segments of the strategy.

        Returns:
            tuple[tuple[int, ScheduleStrategy]]: The segments of the strategy.
        """
        return self._segments

    def __eq__(self, other: object) -> bool:
        """
        Check if two PiecewiseScheduleStrategies are equal.

        Args:
            other (object): The object to compare with.

        Returns:
            bool: True if the strategies are equal, False otherwise.
        """
        # type validations
        if not isinstance(other, PiecewiseScheduleStrategy):
            return NotImplemented
        # equality check
        result = self._segments == other._segments
        return result

    def __hash__(self) -> int:
        """
        Get the hash of the strategy.

        Returns:
            int: The hash of the strategy.
        """
        result = hash((self._segments,))
        return result

    def to_dict(self) -> dict[str, Any]:
        """
        Get the dictionary representation of the strategy.

        Returns:
            dict[str, Any]: The dictionary representation of the strategy.
        """
        segments = tuple(
            (depth, strategy.to_dict()) for depth, strategy in self._segments
        )
        result = {
            "type": self.__class__.__name__,
            "segments": segments,
        }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PiecewiseScheduleStrategy:
        """
        Create a PiecewiseScheduleStrategy from a dictionary.

        Args:
            data (dict[str, Any]): The dictionary representation of the strategy.

        Returns:
            PiecewiseScheduleStrategy: The PiecewiseScheduleStrategy instance.
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
            (depth, ScheduleStrategy.from_dict(strategy_data))
            for depth, strategy_data in data["segments"]
        )
        result = cls(segments)
        return result

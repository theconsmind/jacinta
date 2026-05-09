from __future__ import annotations

import math
from typing import Any

from .ScheduleStrategy import ScheduleStrategy


class LogarithmicScheduleStrategy(ScheduleStrategy):
    """
    A ScheduleStrategy that returns a logarithmic value for a given node depth.

    Attributes:
        scale (float): The scale of the logarithmic function.
        offset (float): The offset of the logarithmic function.
        intercept (float): The intercept of the logarithmic function.
        min_value (float | None): The minimum value of the strategy.
        max_value (float | None): The maximum value of the strategy.
    """

    __slots__ = ("_scale", "_offset", "_intercept", "_min_value", "_max_value")

    def __init__(
        self,
        scale: float,
        offset: float,
        intercept: float,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> None:
        """
        Initialize a LogarithmicScheduleStrategy.

        Args:
            scale (float): The scale of the logarithmic function.
            offset (float): The offset of the logarithmic function.
            intercept (float): The intercept of the logarithmic function.
            min_value (float | None): The minimum value of the strategy.
            max_value (float | None): The maximum value of the strategy.
        """
        # scale validations
        if not isinstance(scale, (float, int)):
            raise TypeError("scale must be a float.")
        # offset validations
        if not isinstance(offset, (float, int)):
            raise TypeError("offset must be a float.")
        # intercept validations
        if not isinstance(intercept, (float, int)):
            raise TypeError("intercept must be a float.")
        # min_value validations
        if min_value is not None and not isinstance(min_value, (float, int)):
            raise TypeError("min_value must be a float or None.")
        # max_value validations
        if max_value is not None and not isinstance(max_value, (float, int)):
            raise TypeError("max_value must be a float or None.")
        if min_value is not None and max_value is not None and min_value > max_value:
            raise ValueError("min_value must be less than or equal to max_value.")
        # initializations
        super().__setattr__("_frozen", False)
        self._scale = float(scale)
        self._offset = float(offset)
        self._intercept = float(intercept)
        self._min_value = float(min_value) if min_value is not None else None
        self._max_value = float(max_value) if max_value is not None else None
        super().__setattr__("_frozen", True)
        return

    def __repr__(self) -> str:
        """
        Get the representation of the strategy.

        Returns:
            str: The representation of the strategy.
        """
        result = (
            f"{self.__class__.__name__}"
            f"(scale={self._scale!r}, offset={self._offset!r}, "
            f"intercept={self._intercept!r}, "
            f"min_value={self._min_value!r}, max_value={self._max_value!r})"
        )
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
        result = self._scale * math.log(depth + self._offset) + self._intercept
        # apply min and max values
        if self._min_value is not None:
            result = max(result, self._min_value)
        if self._max_value is not None:
            result = min(result, self._max_value)
        return result

    @property
    def scale(self) -> float:
        """
        Get the scale of the strategy.

        Returns:
            float: The scale of the strategy.
        """
        return self._scale

    @property
    def offset(self) -> float:
        """
        Get the offset of the strategy.

        Returns:
            float: The offset of the strategy.
        """
        return self._offset

    @property
    def intercept(self) -> float:
        """
        Get the intercept of the strategy.

        Returns:
            float: The intercept of the strategy.
        """
        return self._intercept

    @property
    def min_value(self) -> float | None:
        """
        Get the minimum value of the strategy.

        Returns:
            float | None: The minimum value of the strategy.
        """
        return self._min_value

    @property
    def max_value(self) -> float | None:
        """
        Get the maximum value of the strategy.

        Returns:
            float | None: The maximum value of the strategy.
        """
        return self._max_value

    def __eq__(self, other: object) -> bool:
        """
        Check if two LogarithmicScheduleStrategies are equal.

        Args:
            other (object): The object to compare with.

        Returns:
            bool: True if the strategies are equal, False otherwise.
        """
        # type validations
        if not isinstance(other, LogarithmicScheduleStrategy):
            return NotImplemented
        # equality check
        result = (
            self._scale == other._scale
            and self._offset == other._offset
            and self._intercept == other._intercept
            and self._min_value == other._min_value
            and self._max_value == other._max_value
        )
        return result

    def __hash__(self) -> int:
        """
        Get the hash of the strategy.

        Returns:
            int: The hash of the strategy.
        """
        result = hash(
            (
                self._scale,
                self._offset,
                self._intercept,
                self._min_value,
                self._max_value,
            )
        )
        return result

    def to_dict(self) -> dict[str, Any]:
        """
        Get the dictionary representation of the strategy.

        Returns:
            dict[str, Any]: The dictionary representation of the strategy.
        """
        result = {
            "type": self.__class__.__name__,
            "scale": self._scale,
            "offset": self._offset,
            "intercept": self._intercept,
            "min_value": self._min_value,
            "max_value": self._max_value,
        }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LogarithmicScheduleStrategy:
        """
        Create an LogarithmicScheduleStrategy from a dictionary.

        Args:
            data (dict[str, Any]): The dictionary representation of the strategy.

        Returns:
            LogarithmicScheduleStrategy: The LogarithmicScheduleStrategy instance.
        """
        # data validations
        if not isinstance(data, dict):
            raise TypeError("data must be a dict.")
        if "type" not in data:
            raise KeyError("data must contain the key 'type'.")
        if data["type"] != cls.__name__:
            raise ValueError(f"data['type'] must be a {cls.__name__}.")
        if "scale" not in data:
            raise KeyError("data must contain the key 'scale'.")
        if "offset" not in data:
            raise KeyError("data must contain the key 'offset'.")
        if "intercept" not in data:
            raise KeyError("data must contain the key 'intercept'.")
        if "min_value" not in data:
            raise KeyError("data must contain the key 'min_value'.")
        if "max_value" not in data:
            raise KeyError("data must contain the key 'max_value'.")
        # initializations
        result = cls(
            data["scale"],
            data["offset"],
            data["intercept"],
            data["min_value"],
            data["max_value"],
        )
        return result

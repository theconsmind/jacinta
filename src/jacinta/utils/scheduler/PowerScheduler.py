from __future__ import annotations

import math
from typing import Any

from .Scheduler import Scheduler


class PowerScheduler(Scheduler):
    """
    A Scheduler that assigns a power value to a depth.

    Attributes:
        scale (float): The scale of the power function.
        exponent (float): The exponent of the power function.
        offset (float): The offset of the power function.
        intercept (float): The intercept of the power function.
        min_value (float | None): The minimum value of the scheduler.
        max_value (float | None): The maximum value of the scheduler.
    """

    __slots__ = (
        "_scale",
        "_exponent",
        "_offset",
        "_intercept",
        "_min_value",
        "_max_value",
    )

    def __init__(
        self,
        scale: float,
        exponent: float,
        offset: float,
        intercept: float,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> None:
        """
        Initialize a PowerScheduler.

        Args:
            scale (float): The scale of the power function.
            exponent (float): The exponent of the power function.
            offset (float): The offset of the power function.
            intercept (float): The intercept of the power function.
            min_value (float | None): The minimum value of the scheduler.
                Defaults to None.
            max_value (float | None): The maximum value of the scheduler.
                Defaults to None.
        """
        # scale validations
        if not isinstance(scale, (float, int)):
            raise TypeError("scale must be a float.")
        # exponent validations
        if not isinstance(exponent, (float, int)):
            raise TypeError("exponent must be a float.")
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
        object.__setattr__(self, "_frozen", False)
        self._scale = float(scale)
        self._exponent = float(exponent)
        self._offset = float(offset)
        self._intercept = float(intercept)
        self._min_value = float(min_value) if min_value is not None else None
        self._max_value = float(max_value) if max_value is not None else None
        object.__setattr__(self, "_frozen", True)
        return

    def __repr__(self) -> str:
        """
        Get the representation of the scheduler.

        Returns:
            str: The representation of the scheduler.
        """
        result = (
            f"{self.__class__.__name__}"
            f"(scale={self._scale!r}, exponent={self._exponent!r}, "
            f"offset={self._offset!r}, intercept={self._intercept!r}, "
            f"min_value={self._min_value!r}, max_value={self._max_value!r})"
        )
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
        result = (
            self._scale * math.pow(depth + self._offset, self._exponent)
            + self._intercept
        )
        # apply min and max values
        if self._min_value is not None:
            result = max(result, self._min_value)
        if self._max_value is not None:
            result = min(result, self._max_value)
        return result

    @property
    def scale(self) -> float:
        """
        Get the scale of the scheduler.

        Returns:
            float: The scale of the scheduler.
        """
        return self._scale

    @property
    def exponent(self) -> float:
        """
        Get the exponent of the scheduler.

        Returns:
            float: The exponent of the scheduler.
        """
        return self._exponent

    @property
    def offset(self) -> float:
        """
        Get the offset of the scheduler.

        Returns:
            float: The offset of the scheduler.
        """
        return self._offset

    @property
    def intercept(self) -> float:
        """
        Get the intercept of the scheduler.

        Returns:
            float: The intercept of the scheduler.
        """
        return self._intercept

    @property
    def min_value(self) -> float | None:
        """
        Get the minimum value of the scheduler.

        Returns:
            float | None: The minimum value of the scheduler.
        """
        return self._min_value

    @property
    def max_value(self) -> float | None:
        """
        Get the maximum value of the scheduler.

        Returns:
            float | None: The maximum value of the scheduler.
        """
        return self._max_value

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
        result = (
            self._scale == other._scale
            and self._exponent == other._exponent
            and self._offset == other._offset
            and self._intercept == other._intercept
            and self._min_value == other._min_value
            and self._max_value == other._max_value
        )
        return result

    def to_dict(self) -> dict[str, Any]:
        """
        Get the dictionary representation of the scheduler.

        Returns:
            dict[str, Any]: The dictionary representation of the scheduler.
        """
        result = {
            "type": self.__class__.__name__,
            "scale": self._scale,
            "exponent": self._exponent,
            "offset": self._offset,
            "intercept": self._intercept,
            "min_value": self._min_value,
            "max_value": self._max_value,
        }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PowerScheduler:
        """
        Create a scheduler from a dictionary.

        Args:
            data (dict[str, Any]): The dictionary representation of the scheduler.

        Returns:
            PowerScheduler: The scheduler.
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
        if "exponent" not in data:
            raise KeyError("data must contain the key 'exponent'.")
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
            data["exponent"],
            data["offset"],
            data["intercept"],
            data["min_value"],
            data["max_value"],
        )
        return result

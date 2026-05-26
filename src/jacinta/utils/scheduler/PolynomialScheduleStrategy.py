from __future__ import annotations

import math
from typing import Any

from .ScheduleStrategy import ScheduleStrategy


class PolynomialScheduleStrategy(ScheduleStrategy):
    """
    A ScheduleStrategy that returns a polynomial value for a given depth.

    Attributes:
        coefficients (tuple[float, ...]): The coefficients of the polynomial function.
        min_value (float | None): The minimum value of the PolynomialScheduleStrategy.
        max_value (float | None): The maximum value of the PolynomialScheduleStrategy.
    """

    __slots__ = ("_coefficients", "_min_value", "_max_value")

    def __init__(
        self,
        coefficients: tuple[float, ...],
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> None:
        """
        Initialize a PolynomialScheduleStrategy.

        Args:
            coefficients (tuple[float, ...]): The coefficients
                of the polynomial function.
            min_value (float | None): The minimum value
                of the PolynomialScheduleStrategy.
            max_value (float | None): The maximum value
                of the PolynomialScheduleStrategy.
        """
        # coefficients validations
        if not isinstance(coefficients, (tuple, list)):
            raise TypeError("coefficients must be a tuple.")
        if len(coefficients) == 0:
            raise ValueError("coefficients must not be empty.")
        for coefficient in coefficients:
            if not isinstance(coefficient, (float, int)):
                raise TypeError("All coefficients must be floats.")
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
        self._coefficients = tuple(float(coefficient) for coefficient in coefficients)
        self._min_value = float(min_value) if min_value is not None else None
        self._max_value = float(max_value) if max_value is not None else None
        super().__setattr__("_frozen", True)
        return

    def __repr__(self) -> str:
        """
        Get the representation of the PolynomialScheduleStrategy.

        Returns:
            str: The representation of the PolynomialScheduleStrategy.
        """
        result = (
            f"{self.__class__.__name__}"
            f"(coefficients={self._coefficients!r}, "
            f"min_value={self._min_value!r}, max_value={self._max_value!r})"
        )
        return result

    def __call__(self, depth: int) -> float:
        """
        Get the PolynomialScheduleStrategy value based on the depth.

        Args:
            depth (int): The depth.

        Returns:
            float: The PolynomialScheduleStrategy value based on the depth.
        """
        # depth validations
        if not isinstance(depth, int):
            raise TypeError("depth must be an int.")
        if depth < 0:
            raise ValueError("depth must be greater than or equal to 0.")
        # get the value based on the depth
        result = sum(
            coefficient * math.pow(depth, exponent)
            for exponent, coefficient in enumerate(self._coefficients)
        )
        # apply min and max values
        if self._min_value is not None:
            result = max(result, self._min_value)
        if self._max_value is not None:
            result = min(result, self._max_value)
        return result

    @property
    def coefficients(self) -> tuple[float, ...]:
        """
        Get the coefficients of the PolynomialScheduleStrategy.

        Returns:
            tuple[float, ...]: The coefficients of the PolynomialScheduleStrategy.
        """
        return self._coefficients

    @property
    def min_value(self) -> float | None:
        """
        Get the minimum value of the PolynomialScheduleStrategy.

        Returns:
            float | None: The minimum value of the PolynomialScheduleStrategy.
        """
        return self._min_value

    @property
    def max_value(self) -> float | None:
        """
        Get the maximum value of the PolynomialScheduleStrategy.

        Returns:
            float | None: The maximum value of the PolynomialScheduleStrategy.
        """
        return self._max_value

    def __eq__(self, other: object) -> bool:
        """
        Check if two PolynomialScheduleStrategies are equal.

        Args:
            other (object): The object to compare with.

        Returns:
            bool: True if the PolynomialScheduleStrategies are equal, False otherwise.
        """
        # type validations
        if not isinstance(other, PolynomialScheduleStrategy):
            return NotImplemented
        # equality check
        result = (
            self._coefficients == other._coefficients
            and self._min_value == other._min_value
            and self._max_value == other._max_value
        )
        return result

    def __hash__(self) -> int:
        """
        Get the hash of the PolynomialScheduleStrategy.

        Returns:
            int: The hash of the PolynomialScheduleStrategy.
        """
        result = hash(
            (
                self._coefficients,
                self._min_value,
                self._max_value,
            )
        )
        return result

    def to_dict(self) -> dict[str, Any]:
        """
        Get the dictionary representation of the PolynomialScheduleStrategy.

        Returns:
            dict[str, Any]: The dictionary representation
                of the PolynomialScheduleStrategy.
        """
        result = {
            "type": self.__class__.__name__,
            "coefficients": self._coefficients,
            "min_value": self._min_value,
            "max_value": self._max_value,
        }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolynomialScheduleStrategy:
        """
        Create a PolynomialScheduleStrategy from a dictionary.

        Args:
            data (dict[str, Any]): The dictionary representation
                of the PolynomialScheduleStrategy.

        Returns:
            PolynomialScheduleStrategy: The PolynomialScheduleStrategy instance.
        """
        # data validations
        if not isinstance(data, dict):
            raise TypeError("data must be a dict.")
        if "type" not in data:
            raise KeyError("data must contain the key 'type'.")
        if data["type"] != cls.__name__:
            raise ValueError(f"data['type'] must be a {cls.__name__}.")
        if "coefficients" not in data:
            raise KeyError("data must contain the key 'coefficients'.")
        if "min_value" not in data:
            raise KeyError("data must contain the key 'min_value'.")
        if "max_value" not in data:
            raise KeyError("data must contain the key 'max_value'.")
        # initializations
        result = cls(
            data["coefficients"],
            data["min_value"],
            data["max_value"],
        )
        return result

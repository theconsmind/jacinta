from __future__ import annotations

from typing import Any

from .ScheduleStrategy import ScheduleStrategy


class LinearScheduleStrategy(ScheduleStrategy):
    """
    A ScheduleStrategy that returns a linear value for a given depth.

    Attributes:
        slope (float): The slope of the linear function.
        intercept (float): The intercept of the linear function.
        min_value (float | None): The minimum value of the LinearScheduleStrategy.
        max_value (float | None): The maximum value of the LinearScheduleStrategy.
    """

    __slots__ = ("_slope", "_intercept", "_min_value", "_max_value")

    def __init__(
        self,
        slope: float,
        intercept: float,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> None:
        """
        Initialize a LinearScheduleStrategy.

        Args:
            slope (float): The slope of the linear function.
            intercept (float): The intercept of the linear function.
            min_value (float | None): The minimum value of the LinearScheduleStrategy.
                Defaults to None.
            max_value (float | None): The maximum value of the LinearScheduleStrategy.
                Defaults to None.
        """
        # slope validations
        if not isinstance(slope, (float, int)):
            raise TypeError("slope must be a float.")
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
        self._slope = float(slope)
        self._intercept = float(intercept)
        self._min_value = float(min_value) if min_value is not None else None
        self._max_value = float(max_value) if max_value is not None else None
        super().__setattr__("_frozen", True)
        return

    def __repr__(self) -> str:
        """
        Get the representation of the LinearScheduleStrategy.

        Returns:
            str: The representation of the LinearScheduleStrategy.
        """
        result = (
            f"{self.__class__.__name__}"
            f"(slope={self._slope!r}, intercept={self._intercept!r}, "
            f"min_value={self._min_value!r}, max_value={self._max_value!r})"
        )
        return result

    def __call__(self, depth: int) -> float:
        """
        Get the LinearScheduleStrategy value based on the depth.

        Args:
            depth (int): The depth.

        Returns:
            float: The LinearScheduleStrategy value based on the depth.
        """
        # depth validations
        if not isinstance(depth, int):
            raise TypeError("depth must be an int.")
        if depth < 0:
            raise ValueError("depth must be greater than or equal to 0.")
        # get the value based on the depth
        result = self._slope * depth + self._intercept
        # apply min and max values
        if self._min_value is not None:
            result = max(result, self._min_value)
        if self._max_value is not None:
            result = min(result, self._max_value)
        return result

    @property
    def slope(self) -> float:
        """
        Get the slope of the LinearScheduleStrategy.

        Returns:
            float: The slope of the LinearScheduleStrategy.
        """
        return self._slope

    @property
    def intercept(self) -> float:
        """
        Get the intercept of the LinearScheduleStrategy.

        Returns:
            float: The intercept of the LinearScheduleStrategy.
        """
        return self._intercept

    @property
    def min_value(self) -> float | None:
        """
        Get the minimum value of the LinearScheduleStrategy.

        Returns:
            float | None: The minimum value of the LinearScheduleStrategy.
        """
        return self._min_value

    @property
    def max_value(self) -> float | None:
        """
        Get the maximum value of the LinearScheduleStrategy.

        Returns:
            float | None: The maximum value of the LinearScheduleStrategy.
        """
        return self._max_value

    def __eq__(self, other: object) -> bool:
        """
        Check if two LinearScheduleStrategies are equal.

        Args:
            other (object): The object to compare with.

        Returns:
            bool: True if the LinearScheduleStrategies are equal, False otherwise.
        """
        # other validations
        if not isinstance(other, LinearScheduleStrategy):
            return NotImplemented
        # equality check
        result = (
            self._slope == other._slope
            and self._intercept == other._intercept
            and self._min_value == other._min_value
            and self._max_value == other._max_value
        )
        return result

    def to_dict(self) -> dict[str, Any]:
        """
        Get the dictionary representation of the LinearScheduleStrategy.

        Returns:
            dict[str, Any]: The dictionary representation of the LinearScheduleStrategy.
        """
        result = {
            "type": self.__class__.__name__,
            "slope": self._slope,
            "intercept": self._intercept,
            "min_value": self._min_value,
            "max_value": self._max_value,
        }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LinearScheduleStrategy:
        """
        Create a LinearScheduleStrategy from a dictionary.

        Args:
            data (dict[str, Any]): The dictionary representation
                of the LinearScheduleStrategy.

        Returns:
            LinearScheduleStrategy: The LinearScheduleStrategy instance.
        """
        # data validations
        if not isinstance(data, dict):
            raise TypeError("data must be a dict.")
        if "type" not in data:
            raise KeyError("data must contain the key 'type'.")
        if data["type"] != cls.__name__:
            raise ValueError(f"data['type'] must be a {cls.__name__}.")
        if "slope" not in data:
            raise KeyError("data must contain the key 'slope'.")
        if "intercept" not in data:
            raise KeyError("data must contain the key 'intercept'.")
        if "min_value" not in data:
            raise KeyError("data must contain the key 'min_value'.")
        if "max_value" not in data:
            raise KeyError("data must contain the key 'max_value'.")
        # initializations
        result = cls(
            data["slope"],
            data["intercept"],
            data["min_value"],
            data["max_value"],
        )
        return result

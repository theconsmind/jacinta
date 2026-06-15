from __future__ import annotations

from typing import Any

from .Scheduler import Scheduler


class ConstantScheduler(Scheduler):
    """
    A Scheduler that assigns a constant value to a depth.

    Attributes:
        value (float): The constant value of the scheduler.
    """

    __slots__ = ("_value",)

    def __init__(self, value: float) -> None:
        """
        Initialize a ConstantScheduler.

        Args:
            value (float): The value of the scheduler.
        """
        # value validations
        if not isinstance(value, (float, int)):
            raise TypeError("value must be a float.")
        # initializations
        object.__setattr__(self, "_frozen", False)
        self._value = float(value)
        object.__setattr__(self, "_frozen", True)
        return

    def __repr__(self) -> str:
        """
        Get the representation of the scheduler.

        Returns:
            str: The representation of the scheduler.
        """
        result = f"{self.__class__.__name__}(value={self._value})"
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
        result = self._value
        return result

    @property
    def value(self) -> float:
        """
        Get the value of the scheduler.

        Returns:
            float: The value of the scheduler.
        """
        return self._value

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
        result = self._value == other._value
        return result

    def to_dict(self) -> dict[str, Any]:
        """
        Get the dictionary representation of the scheduler.

        Returns:
            dict[str, Any]: The dictionary representation of the scheduler.
        """
        result = {
            "type": self.__class__.__name__,
            "value": self._value,
        }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConstantScheduler:
        """
        Create a scheduler from a dictionary.

        Args:
            data (dict[str, Any]): The dictionary representation of the scheduler.

        Returns:
            ConstantScheduler: The scheduler.
        """
        # data validations
        if not isinstance(data, dict):
            raise TypeError("data must be a dict.")
        if "type" not in data:
            raise KeyError("data must contain the key 'type'.")
        if data["type"] != cls.__name__:
            raise ValueError(f"data['type'] must be a {cls.__name__}.")
        if "value" not in data:
            raise KeyError("data must contain the key 'value'.")
        # initializations
        result = cls(data["value"])
        return result

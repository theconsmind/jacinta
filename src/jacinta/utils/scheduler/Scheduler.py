from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .ScheduleStrategy import ScheduleStrategy


class Scheduler:
    """
    A Scheduler represents a callable scheduler that maps a node depth to
    a value using a schedule strategy.

    Attributes:
        strategy (ScheduleStrategy): The schedule strategy to use.
    """

    __slots__ = ("_strategy", "_frozen")

    def __init__(self, strategy: ScheduleStrategy) -> None:
        """
        Initialize the Scheduler.

        Args:
            strategy (ScheduleStrategy): The schedule strategy to use.
        """
        # strategy validations
        if not isinstance(strategy, ScheduleStrategy):
            raise TypeError("strategy must be an instance of ScheduleStrategy.")
        # initializations
        super().__setattr__("_frozen", False)
        self._strategy = strategy
        super().__setattr__("_frozen", True)
        return

    def __repr__(self) -> str:
        """
        Get the representation of the scheduler.

        Returns:
            str: The representation of the scheduler.
        """
        result = f"Scheduler(strategy={self._strategy!r})"
        return result

    def __call__(self, depth: int) -> float:
        """
        Get the schedule value based on the node depth.

        Args:
            depth (int): The depth of the node.

        Returns:
            float: The schedule value based on the node depth.
        """
        # depth validations
        if not isinstance(depth, int):
            raise TypeError("depth must be an int.")
        if depth < 0:
            raise ValueError("depth must be greater than or equal to 0.")
        # get the value based on the depth
        result = self._strategy(depth)
        return result

    @property
    def strategy(self) -> ScheduleStrategy:
        """
        Get the schedule strategy of the scheduler.

        Returns:
            ScheduleStrategy: The schedule strategy of the scheduler.
        """
        return self._strategy

    def __eq__(self, other: object) -> bool:
        """
        Check if two Schedulers are equal.

        Args:
            other (object): The object to compare with.

        Returns:
            bool: True if the schedulers are equal, False otherwise.
        """
        if not isinstance(other, Scheduler):
            return NotImplemented
        result = self._strategy == other._strategy
        return result

    def __hash__(self) -> int:
        """
        Get the hash of the scheduler.

        Returns:
            int: The hash of the scheduler.
        """
        result = hash(self._strategy)
        return result

    def to_dict(self) -> dict[str, Any]:
        """
        Get the dictionary representation of the scheduler.

        Returns:
            dict[str, Any]: The dictionary representation of the scheduler.
        """
        result = {
            "type": self.__class__.__name__,
            "strategy": self._strategy.to_dict(),
        }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scheduler:
        """
        Create a Scheduler from a dictionary.

        Args:
            data (dict[str, Any]): The dictionary representation of the scheduler.

        Returns:
            Scheduler: The Scheduler instance.
        """
        # data validations
        if not isinstance(data, dict):
            raise TypeError("data must be a dict.")
        if "type" not in data:
            raise KeyError("data must contain the key 'type'.")
        if data["type"] != cls.__name__:
            raise ValueError(f"data['type'] must be a {cls.__name__}.")
        if "strategy" not in data:
            raise KeyError("data must contain the key 'strategy'.")
        # initializations
        result = cls(ScheduleStrategy.from_dict(data["strategy"]))
        return result

    def save(self, path: str | Path, overwrite: bool = False) -> None:
        """
        Save the scheduler to a json file.

        Args:
            path (str | Path): The path to the file.
            overwrite (bool): Whether to overwrite the file if it exists.
        """
        # path validations
        if not isinstance(path, (str, Path)):
            raise TypeError("path must be a string or a Path.")
        # file validations
        path = Path(path)
        if path.suffix != ".json":
            raise ValueError("path must have a .json extension.")
        if not overwrite and path.exists():
            raise FileExistsError(f"path already exists: {path}")
        # file creation
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=4)
        return

    @classmethod
    def load(cls, path: str | Path) -> Scheduler:
        """
        Load the scheduler from a json file.

        Args:
            path (str | Path): The path to the file.

        Returns:
            Scheduler: The Scheduler instance.
        """
        # path validations
        if not isinstance(path, (str, Path)):
            raise TypeError("path must be a string or a Path.")
        # file validations
        path = Path(path)
        if path.suffix != ".json":
            raise ValueError("path must have a .json extension.")
        if not path.exists():
            raise FileNotFoundError(f"path does not exist: {path}")
        # file loading
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        result = cls.from_dict(data)
        return result

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Set an attribute of the scheduler.

        Args:
            name (str): The name of the attribute.
            value (Any): The value of the attribute.
        """
        if getattr(self, "_frozen", False):
            raise AttributeError(f"{self.__class__.__name__} is immutable")
        super().__setattr__(name, value)
        return

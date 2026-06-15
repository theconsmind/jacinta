from __future__ import annotations

import json
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Self


class Scheduler(ABC):
    """
    A Scheduler represents a strategy that assigns a value to a depth.
    """

    __slots__ = ("_frozen",)

    @abstractmethod
    def __call__(self, depth: int) -> float:
        """
        Get the value assigned to the given depth.

        Args:
            depth (int): The depth.

        Returns:
            float: The value assigned to the given depth.
        """
        ...

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """
        Check if two schedulers are equal.

        Args:
            other (object): The object to compare with.

        Returns:
            bool: True if the schedulers are equal, False otherwise.
        """
        ...

    def copy(self) -> Self:
        """
        Get a copy of the scheduler.

        Returns:
            Self: The copy of the scheduler.
        """
        result = deepcopy(self)
        return result

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """
        Get the dictionary representation of the scheduler.

        Returns:
            dict[str, Any]: The dictionary representation of the scheduler.
        """
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """
        Create a scheduler from a dictionary.

        Args:
            data (dict[str, Any]): The dictionary representation of the scheduler.

        Returns:
            Self: The scheduler.
        """
        # data validations
        if not isinstance(data, dict):
            raise TypeError("data must be a dict.")
        if "type" not in data:
            raise KeyError("data must contain the key 'type'.")
        if not isinstance(data["type"], str):
            raise TypeError("data['type'] must be a string.")
        # find the subclass
        result = None
        for subclass in cls.__subclasses__():
            if subclass.__name__ == data["type"]:
                result = subclass.from_dict(data)
                break
        if result is None:
            raise ValueError(f"Scheduler type '{data['type']}' not found.")
        return result

    def save(self, path: str | Path, overwrite: bool = False) -> None:
        """
        Save the scheduler to a json file.

        Args:
            path (str | Path): The path to the file.
            overwrite (bool): Whether to overwrite the file if it exists.
                Defaults to False.
        """
        # path validations
        if not isinstance(path, (str, Path)):
            raise TypeError("path must be a string or a Path.")
        # file validations
        path = Path(path)
        if path.suffix != ".json":
            raise ValueError("path must have a .json extension.")
        if not overwrite and path.exists():
            raise FileExistsError(f"path already exists: {path}.")
        # file creation
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=4)
        return

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """
        Load a scheduler from a json file.

        Args:
            path (str | Path): The path to the file.

        Returns:
            Self: The scheduler.
        """
        # path validations
        if not isinstance(path, (str, Path)):
            raise TypeError("path must be a string or a Path.")
        # file validations
        path = Path(path)
        if path.suffix != ".json":
            raise ValueError("path must have a .json extension.")
        if not path.exists():
            raise FileNotFoundError(f"path does not exist: {path}.")
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
        # freeze check
        if getattr(self, "_frozen", False):
            raise AttributeError(f"{self.__class__.__name__} is immutable.")
        # set the attribute
        object.__setattr__(self, name, value)
        return

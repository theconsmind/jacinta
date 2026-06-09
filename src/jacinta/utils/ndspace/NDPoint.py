from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Self


class NDPoint:
    """
    An NDPoint represents an N-dimensional point.

    Attributes:
        coordinates (tuple[float, ...]): The coordinates of the point.
    """

    __slots__ = ("_coordinates", "_frozen")

    def __init__(self, coordinates: tuple[float, ...]) -> None:
        """
        Initialize an NDPoint.

        Args:
            coordinates (tuple[float, ...]): The coordinates of the point.
        """
        # coordinates validations
        if not isinstance(coordinates, (tuple, list)):
            raise TypeError("coordinates must be a tuple.")
        for coord in coordinates:
            if not isinstance(coord, (float, int)):
                raise TypeError("All coordinates must be floats.")
        # initializations
        object.__setattr__(self, "_frozen", False)
        self._coordinates = tuple(float(coord) for coord in coordinates)
        object.__setattr__(self, "_frozen", True)
        return

    def __repr__(self) -> str:
        """
        Get the representation of the point.

        Returns:
            str: The representation of the point.
        """
        result = f"{self.__class__.__name__}(coordinates={self._coordinates!r})"
        return result

    @property
    def coordinates(self) -> tuple[float, ...]:
        """
        Get the coordinates of the point.

        Returns:
            tuple[float, ...]: The coordinates of the point.
        """
        return self._coordinates

    @property
    def nd(self) -> int:
        """
        Get the number of dimensions of the point.

        Returns:
            int: The number of dimensions of the point.
        """
        nd = len(self._coordinates)
        return nd

    def __eq__(self, other: object) -> bool:
        """
        Check if two points are equal.

        Args:
            other (object): The object to compare with.

        Returns:
            bool: True if the points are equal, False otherwise.
        """
        # other validations
        if type(self) is not type(other):
            return NotImplemented
        # equality check
        result = self._coordinates == other._coordinates
        return result

    def copy(self) -> Self:
        """
        Get a copy of the point.

        Returns:
            Self: A copy of the point.
        """
        result = deepcopy(self)
        return result

    def to_dict(self) -> dict[str, Any]:
        """
        Get the dictionary representation of the point.

        Returns:
            dict[str, Any]: The dictionary representation of the point.
        """
        result = {
            "type": self.__class__.__name__,
            "coordinates": self._coordinates,
        }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """
        Create a point from a dictionary.

        Args:
            data (dict[str, Any]): The dictionary representation of the point.

        Returns:
            Self: The point.
        """
        # data validations
        if not isinstance(data, dict):
            raise TypeError("data must be a dict.")
        if "type" not in data:
            raise KeyError("data must contain the key 'type'.")
        if data["type"] != cls.__name__:
            raise ValueError(f"data['type'] must be a {cls.__name__}.")
        if "coordinates" not in data:
            raise KeyError("data must contain the key 'coordinates'.")
        # initializations
        result = cls(data["coordinates"])
        return result

    def save(self, path: str | Path, overwrite: bool = False) -> None:
        """
        Save the point to a json file.

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
        Load a point from a json file.

        Args:
            path (str | Path): The path to the file.

        Returns:
            Self: The point.
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
        Set an attribute of the point.

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

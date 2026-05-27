from __future__ import annotations

import json
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

from .NDPoint import NDPoint


class NDSpace:
    """
    An NDSpace represents an N-dimensional space.

    Attributes:
        bounds (tuple[tuple[float, float], ...]): The bounds of the NDSpace.
    """

    __slots__ = ("_bounds", "_frozen")

    def __init__(self, bounds: tuple[tuple[float, float], ...]) -> None:
        """
        Initialize an NDSpace.

        Args:
            bounds (tuple[tuple[float, float], ...]): The bounds of the NDSpace.
        """
        # bounds validations
        if not isinstance(bounds, (tuple, list)):
            raise TypeError("bounds must be a tuple.")
        for bound in bounds:
            if not isinstance(bound, (tuple, list)):
                raise TypeError("All bounds must be tuples.")
            if len(bound) != 2:
                raise ValueError("All bounds must have length 2.")
            if not isinstance(bound[0], (float, int)):
                raise TypeError("All lower bounds must be floats.")
            if not isinstance(bound[1], (float, int)):
                raise TypeError("All upper bounds must be floats.")
            if bound[0] >= bound[1]:
                raise ValueError(
                    "All lower bounds must be lower than their respective upper bounds."
                )
        # initializations
        super().__setattr__("_frozen", False)
        self._bounds = tuple((float(lower), float(upper)) for (lower, upper) in bounds)
        super().__setattr__("_frozen", True)
        return

    def __repr__(self) -> str:
        """
        Get the representation of the NDSpace.

        Returns:
            str: The representation of the NDSpace.
        """
        result = f"{self.__class__.__name__}(bounds={self._bounds})"
        return result

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        """
        Get the bounds of the NDSpace.

        Returns:
            tuple[tuple[float, float], ...]: The bounds of the NDSpace.
        """
        return self._bounds

    @property
    def nd(self) -> int:
        """
        Get the number of dimensions of the NDSpace.

        Returns:
            int: The number of dimensions of the NDSpace.
        """
        nd = len(self._bounds)
        return nd

    def __eq__(self, other: object) -> bool:
        """
        Check if two NDSpaces are equal.

        Args:
            other (object): The object to compare with.

        Returns:
            bool: True if the NDSpaces are equal, False otherwise.
        """
        # type validations
        if not isinstance(other, NDSpace):
            return NotImplemented
        # equality check
        result = self._bounds == other._bounds
        return result

    def __hash__(self) -> int:
        """
        Get the hash of the NDSpace.

        Returns:
            int: The hash of the NDSpace.
        """
        result = hash((self._bounds,))
        return result

    def contains(self, point: NDPoint) -> bool:
        """
        Check if an NDPoint is within the bounds of the NDSpace.

        Args:
            point (NDPoint): The NDPoint to check.

        Returns:
            bool: True if the NDPoint is within the bounds, False otherwise.
        """
        # point validations
        if not isinstance(point, NDPoint):
            raise TypeError("point must be an NDPoint.")
        if point.nd != self.nd:
            raise ValueError(f"point must be {self.nd}D.")
        # check if the point is within the bounds
        contains = all(
            lower <= coord < upper
            for coord, (lower, upper) in zip(
                point.coordinates, self._bounds, strict=True
            )
        )
        return contains

    def split(self, point: NDPoint) -> tuple[NDSpace, ...]:
        """
        Split the NDSpace into smaller NDSpaces based on an NDPoint.

        Args:
            point (NDPoint): The NDPoint to split the NDSpace by.

        Returns:
            tuple[NDSpace, ...]: The sub-NDSpaces created by the split.
        """
        # point validations
        if not isinstance(point, NDPoint):
            raise TypeError("point must be an NDPoint.")
        if point.nd != self.nd:
            raise ValueError(f"point must be {self.nd}D.")
        if not self.contains(point):
            raise ValueError("point must be in range.")
        # split the space
        ndspaces = []
        # generate all combinations of upper/lower halves
        for directions in product((False, True), repeat=self.nd):
            new_bounds = list(self._bounds)
            is_valid = True
            # build bounds for each subspace
            for dim, upper_half in enumerate(directions):
                lower, upper = self._bounds[dim]
                if upper_half:
                    new_bound = (point.coordinates[dim], upper)
                else:
                    new_bound = (lower, point.coordinates[dim])
                # skip if the new bound is empty (lower == upper)
                if new_bound[0] == new_bound[1]:
                    is_valid = False
                    break
                new_bounds[dim] = new_bound
            # create new ndspace if valid (lower < upper)
            if is_valid:
                ndspace = NDSpace(tuple(new_bounds))
                ndspaces.append(ndspace)
        ndspaces = tuple(ndspaces)
        return ndspaces

    def add_dimensions(self, bounds: tuple[tuple[float, float], ...]) -> NDSpace:
        """
        Add new dimensions to the NDSpace.

        Args:
            bounds (tuple[tuple[float, float], ...]): The bounds of the new dimensions.

        Returns:
            NDSpace: The new NDSpace with added dimensions.
        """
        # bounds validations
        if not isinstance(bounds, (tuple, list)):
            raise TypeError("bounds must be a tuple.")
        for bound in bounds:
            if not isinstance(bound, (tuple, list)):
                raise TypeError("All bounds must be tuples.")
            if len(bound) != 2:
                raise ValueError("All bounds must have length 2.")
            if not isinstance(bound[0], (float, int)):
                raise TypeError("All lower bounds must be floats.")
            if not isinstance(bound[1], (float, int)):
                raise TypeError("All upper bounds must be floats.")
            if bound[0] >= bound[1]:
                raise ValueError(
                    "All lower bounds must be lower than their respective upper bounds."
                )
        # create new NDSpace with new bound
        new_bounds = tuple((float(lower), float(upper)) for (lower, upper) in bounds)
        new_bounds = self._bounds + new_bounds
        ndspace = NDSpace(new_bounds)
        return ndspace

    def remove_dimensions(self, dims: tuple[int, ...]) -> NDSpace:
        """
        Remove dimensions from the NDSpace.

        Args:
            dims (tuple[int, ...]): The indices of the dimensions to remove.

        Returns:
            NDSpace: The new NDSpace with removed dimensions.
        """
        # dims validations
        if not isinstance(dims, (tuple, list)):
            raise TypeError("dims must be a tuple.")
        for dim in dims:
            if not isinstance(dim, int):
                raise TypeError("All dims must be ints.")
            if not (0 <= dim < self.nd):
                raise IndexError("All dims must be in range.")
        # create new NDSpace without the dimension
        new_bounds = tuple(
            bound for idx, bound in enumerate(self._bounds) if idx not in dims
        )
        ndspace = NDSpace(new_bounds)
        return ndspace

    def copy(self) -> NDSpace:
        """
        Get a copy of the NDSpace.

        Returns:
            NDSpace: A copy of the NDSpace.
        """
        result = deepcopy(self)
        return result

    def to_dict(self) -> dict[str, Any]:
        """
        Get the dictionary representation of the NDSpace.

        Returns:
            dict[str, Any]: The dictionary representation of the NDSpace.
        """
        result = {
            "type": self.__class__.__name__,
            "bounds": self._bounds,
        }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NDSpace:
        """
        Create an NDSpace from a dictionary.

        Args:
            data (dict[str, Any]): The dictionary representation of the NDSpace.

        Returns:
            NDSpace: The NDSpace instance.
        """
        # data validations
        if not isinstance(data, dict):
            raise TypeError("data must be a dict.")
        if "type" not in data:
            raise KeyError("data must contain the key 'type'.")
        if data["type"] != cls.__name__:
            raise ValueError(f"data['type'] must be a {cls.__name__}.")
        if "bounds" not in data:
            raise KeyError("data must contain the key 'bounds'.")
        # initializations
        result = cls(data["bounds"])
        return result

    def save(self, path: str | Path, overwrite: bool = False) -> None:
        """
        Save the NDSpace to a json file.

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
            raise FileExistsError(f"path already exists: {path}.")
        # file creation
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=4)
        return

    @classmethod
    def load(cls, path: str | Path) -> NDSpace:
        """
        Load an NDSpace from a json file.

        Args:
            path (str | Path): The path to the file.

        Returns:
            NDSpace: The NDSpace instance.
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
        Set an attribute of the NDSpace.

        Args:
            name (str): The name of the attribute.
            value (Any): The value of the attribute.
        """
        # freeze check
        if getattr(self, "_frozen", False):
            raise AttributeError(f"{self.__class__.__name__} is immutable.")
        # set the attribute
        super().__setattr__(name, value)
        return

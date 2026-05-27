from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .NDSpace import NDSpace


class NDTree:
    """
    An NDTree represents a tree of N-dimensional spaces.

    Attributes:
        space (NDSpace): The root NDSpace of the NDTree.
        depth (int): The depth of the NDTree.
    """

    __slots__ = (
        "_space",
        "_depth",
        "_frozen",
    )

    def __init__(self, space: NDSpace, depth: int) -> None:
        """
        Initialize an NDTree.

        Args:
            space (NDSpace): The root NDSpace of the NDTree.
        """
        # space validations
        if not isinstance(space, NDSpace):
            raise TypeError("space must be an NDSpace.")
        # depth validations
        if not isinstance(depth, int):
            raise TypeError("depth must be an int.")
        if depth < 0:
            raise ValueError("depth must be greater than or equal to 0.")
        # initializations
        super().__setattr__("_frozen", False)
        self._space = space.copy()
        self._depth = depth
        super().__setattr__("_frozen", True)
        return

    def __repr__(self) -> str:
        """
        Get the representation of the NDTree.

        Returns:
            str: The representation of the NDTree.
        """
        result = (
            f"{self.__class__.__name__}(space={self._space!r}, depth={self._depth!r})"
        )
        return result

    @property
    def space(self) -> NDSpace:
        """
        Get the root NDSpace of the NDTree.

        Returns:
            NDSpace: The root NDSpace of the NDTree.
        """
        return self._space

    @property
    def depth(self) -> int:
        """
        Get the depth of the NDTree.

        Returns:
            int: The depth of the NDTree.
        """
        return self._depth

    @property
    def nd(self) -> int:
        """
        Get the number of dimensions of the NDTree.

        Returns:
            int: The number of dimensions of the NDTree.
        """
        nd = self._space.nd
        return nd

    def __eq__(self, other: object) -> bool:
        """
        Check if two NDTrees are equal.

        Args:
            other (object): The object to compare with.

        Returns:
            bool: True if the NDTrees are equal, False otherwise.
        """
        # type validations
        if not isinstance(other, NDTree):
            return NotImplemented
        # equality check
        result = self._space == other._space and self._depth == other._depth
        return result

    def __hash__(self) -> int:
        """
        Get the hash of the NDTree.

        Returns:
            int: The hash of the NDTree.
        """
        result = hash(
            (
                self._space,
                self._depth,
            )
        )
        return result

    def copy(self) -> NDTree:
        """
        Get a copy of the NDTree.

        Returns:
            NDTree: A copy of the NDTree.
        """
        result = deepcopy(self)
        return result

    def to_dict(self) -> dict[str, Any]:
        """
        Get the dictionary representation of the NDTree.

        Returns:
            dict[str, Any]: The dictionary representation of the NDTree.
        """
        result = {
            "type": self.__class__.__name__,
            "space": self._space,
            "depth": self._depth,
        }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NDTree:
        """
        Create an NDTree from a dictionary.

        Args:
            data (dict[str, Any]): The dictionary representation of the NDTree.

        Returns:
            NDTree: The NDTree instance.
        """
        # data validations
        if not isinstance(data, dict):
            raise TypeError("data must be a dict.")
        if "type" not in data:
            raise KeyError("data must contain the key 'type'.")
        if data["type"] != cls.__name__:
            raise ValueError(f"data['type'] must be a {cls.__name__}.")
        if "space" not in data:
            raise KeyError("data must contain the key 'space'.")
        if "depth" not in data:
            raise KeyError("data must contain the key 'depth'.")
        # initializations
        result = cls(NDSpace.from_dict(data["space"]), data["depth"])
        return result

    def save(self, path: str | Path, overwrite: bool = False) -> None:
        """
        Save the NDTree to a json file.

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
    def load(cls, path: str | Path) -> NDTree:
        """
        Load an NDTree from a json file.

        Args:
            path (str | Path): The path to the file.

        Returns:
            NDTree: The NDTree instance.
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
        Set an attribute of the NDTree.

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

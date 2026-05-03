from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class TransmitterNode:
    """
    A TransmitterNode represents a node in the Transmitter tree.

    Attributes:
        left (float): The closed left bound of the node's interval.
        right (float): The open right bound of the node's interval.
        parent_id (int): The ID of the parent node.
        left_child_id (int): The ID of the left child node.
        right_child_id (int): The ID of the right child node.
        weight (float): The weight of the node.
        mass (float): The probability of the node being selected.
        depth (int): The depth of the node.
        hits_left (int): The number of hits left to split the node.
    """

    __slots__ = (
        "_left",
        "_right",
        "_parent_id",
        "_left_child_id",
        "_right_child_id",
        "_weight",
        "_mass",
        "_depth",
        "_hits_left",
        "_frozen",
    )

    def __init__(
        self,
        left: float,
        right: float,
        parent_id: int,
        left_child_id: int,
        right_child_id: int,
        weight: float,
        mass: float,
        depth: int,
        hits_left: int,
    ) -> None:
        """
        Initialize a TransmitterNode.

        Args:
            left (float): The closed left bound of the node's interval.
            right (float): The open right bound of the node's interval.
            parent_id (int): The ID of the parent node.
            left_child_id (int): The ID of the left child node.
            right_child_id (int): The ID of the right child node.
            weight (float): The weight of the node.
            mass (float): The probability of the node being selected.
            depth (int): The depth of the node.
            hits_left (int): The number of hits left to split the node.
        """
        # left & right validations
        if not isinstance(left, (float, int)):
            raise TypeError("left must be a float.")
        if not isinstance(right, (float, int)):
            raise TypeError("right must be a float.")
        if left >= right:
            raise ValueError("left must be lower than right.")
        # node_ids validations
        if not isinstance(parent_id, int):
            raise TypeError("parent_id must be an int.")
        if parent_id < -1:
            raise ValueError("parent_id must be greater than or equal to -1.")
        if not isinstance(left_child_id, int):
            raise TypeError("left_child_id must be an int.")
        if left_child_id < -1:
            raise ValueError("left_child_id must be greater than or equal to -1.")
        if not isinstance(right_child_id, int):
            raise TypeError("right_child_id must be an int.")
        if right_child_id < -1:
            raise ValueError("right_child_id must be greater than or equal to -1.")
        # weight validations
        if not isinstance(weight, (float, int)):
            raise TypeError("weight must be a float.")
        if weight <= 0:
            raise ValueError("weight must be greater than 0.")
        # mass validations
        if not isinstance(mass, (float, int)):
            raise TypeError("mass must be a float.")
        if mass <= 0:
            raise ValueError("mass must be greater than 0.")
        # depth validations
        if not isinstance(depth, int):
            raise TypeError("depth must be an int.")
        if depth < 0:
            raise ValueError("depth must be greater than or equal to 0.")
        # hits_left validations
        if not isinstance(hits_left, int):
            raise TypeError("hits_left must be an int.")
        if hits_left < 0:
            raise ValueError("hits_left must be greater than or equal to 0.")
        # initializations
        super().__setattr__("_frozen", False)
        self._left = float(left)
        self._right = float(right)
        self._parent_id = parent_id
        self._left_child_id = left_child_id
        self._right_child_id = right_child_id
        self._weight = float(weight)
        self._mass = float(mass)
        self._depth = depth
        self._hits_left = hits_left
        super().__setattr__("_frozen", True)
        return

    def __repr__(self) -> str:
        """
        Get the representation of the node.

        Returns:
            str: The representation of the node.
        """
        result = (
            f"{self.__class__.__name__}"
            f"(left={self._left!r}, right={self._right!r}, "
            f"parent_id={self._parent_id!r}, "
            f"left_child_id={self._left_child_id!r}, "
            f"right_child_id={self._right_child_id!r}, "
            f"weight={self._weight!r}, "
            f"mass={self._mass!r}, "
            f"depth={self._depth!r}, "
            f"hits_left={self._hits_left!r})"
        )
        return result

    @property
    def left(self) -> float:
        """
        Get the closed left bound of the node's interval.

        Returns:
            float: The closed left bound of the node's interval.
        """
        return self._left

    @property
    def right(self) -> float:
        """
        Get the open right bound of the node's interval.

        Returns:
            float: The open right bound of the node's interval.
        """
        return self._right

    @property
    def parent_id(self) -> int:
        """
        Get the ID of the parent node.

        Returns:
            int: The ID of the parent node.
        """
        return self._parent_id

    @property
    def left_child_id(self) -> int:
        """
        Get the ID of the left child node.

        Returns:
            int: The ID of the left child node.
        """
        return self._left_child_id

    @left_child_id.setter
    def left_child_id(self, value: int) -> None:
        """
        Set the ID of the left child node.

        Args:
            value (int): The ID of the left child node.
        """
        if self._left_child_id != -1:
            raise AttributeError("left_child_id already set.")
        # value validations
        if not isinstance(value, int):
            raise TypeError("left_child_id must be an int.")
        if value < -1:
            raise ValueError("left_child_id must be greater than or equal to -1.")
        # update value
        super().__setattr__("_frozen", False)
        self._left_child_id = value
        super().__setattr__("_frozen", True)
        return

    @property
    def right_child_id(self) -> int:
        """
        Get the ID of the right child node.

        Returns:
            int: The ID of the right child node.
        """
        return self._right_child_id

    @right_child_id.setter
    def right_child_id(self, value: int) -> None:
        """
        Set the ID of the right child node.

        Args:
            value (int): The ID of the right child node.
        """
        if self._right_child_id != -1:
            raise AttributeError("right_child_id already set.")
        # value validations
        if not isinstance(value, int):
            raise TypeError("right_child_id must be an int.")
        if value < -1:
            raise ValueError("right_child_id must be greater than or equal to -1.")
        # update value
        super().__setattr__("_frozen", False)
        self._right_child_id = value
        super().__setattr__("_frozen", True)
        return

    @property
    def weight(self) -> float:
        """
        Get the weight of the node.

        Returns:
            float: The weight of the node.
        """
        return self._weight

    @weight.setter
    def weight(self, value: float) -> None:
        """
        Set the weight of the node.

        Args:
            value (float): The weight of the node.
        """
        # value validations
        if not isinstance(value, (float, int)):
            raise TypeError("weight must be a float.")
        if value <= 0:
            raise ValueError("weight must be greater than 0.")
        # update value
        super().__setattr__("_frozen", False)
        self._weight = float(value)
        super().__setattr__("_frozen", True)
        return

    @property
    def mass(self) -> float:
        """
        Get the probability of the node being selected.

        Returns:
            float: The probability of the node being selected.
        """
        return self._mass

    @mass.setter
    def mass(self, value: float) -> None:
        """
        Set the probability of the node being selected.

        Args:
            value (float): The probability of the node being selected.
        """
        # value validations
        if not isinstance(value, (float, int)):
            raise TypeError("mass must be a float.")
        if value <= 0:
            raise ValueError("mass must be greater than 0.")
        # update value
        super().__setattr__("_frozen", False)
        self._mass = float(value)
        super().__setattr__("_frozen", True)
        return

    @property
    def depth(self) -> int:
        """
        Get the depth of the node.

        Returns:
            int: The depth of the node.
        """
        return self._depth

    @property
    def hits_left(self) -> int:
        """
        Get the number of hits left to split the node.

        Returns:
            int: The number of hits left to split the node.
        """
        return self._hits_left

    @hits_left.setter
    def hits_left(self, value: int) -> None:
        """
        Set the number of hits left to split the node.

        Args:
            value (int): The number of hits left to split the node.
        """
        # value validations
        if not isinstance(value, int):
            raise TypeError("hits_left must be an int.")
        if value < 0:
            raise ValueError("hits_left must be greater than or equal to 0.")
        # update value
        super().__setattr__("_frozen", False)
        self._hits_left = value
        super().__setattr__("_frozen", True)
        return

    @property
    def is_leaf(self) -> bool:
        """
        Check if the node is a leaf.

        Returns:
            bool: True if the node is a leaf, False otherwise.
        """
        result = self._left_child_id == -1 and self._right_child_id == -1
        return result

    @property
    def length(self) -> float:
        """
        Get the length of the node's interval.

        Returns:
            float: The length of the node's interval.
        """
        result = self._right - self._left
        return result

    def __eq__(self, other: object) -> bool:
        """
        Check if two TransmitterNodes are equal.

        Args:
            other (object): The object to compare with.

        Returns:
            bool: True if the nodes are equal, False otherwise.
        """
        # type validations
        if not isinstance(other, TransmitterNode):
            return NotImplemented
        # equality check
        result = (
            self._left == other._left
            and self._right == other._right
            and self._parent_id == other._parent_id
            and self._left_child_id == other._left_child_id
            and self._right_child_id == other._right_child_id
            and self._weight == other._weight
            and self._mass == other._mass
            and self._depth == other._depth
            and self._hits_left == other._hits_left
        )
        return result

    def to_dict(self) -> dict[str, Any]:
        """
        Get the dictionary representation of the node.

        Returns:
            dict[str, Any]: The dictionary representation of the node.
        """
        result = {
            "type": self.__class__.__name__,
            "left": self._left,
            "right": self._right,
            "parent_id": self._parent_id,
            "left_child_id": self._left_child_id,
            "right_child_id": self._right_child_id,
            "weight": self._weight,
            "mass": self._mass,
            "depth": self._depth,
            "hits_left": self._hits_left,
        }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransmitterNode:
        """
        Create a TransmitterNode from a dictionary.

        Args:
            data (dict[str, Any]): The dictionary representation of the node.

        Returns:
            TransmitterNode: The TransmitterNode instance.
        """
        # data validations
        if not isinstance(data, dict):
            raise TypeError("data must be a dict.")
        if "type" not in data:
            raise KeyError("data must contain the key 'type'.")
        if data["type"] != cls.__name__:
            raise ValueError(f"data['type'] must be a {cls.__name__}.")
        if "left" not in data:
            raise KeyError("data must contain the key 'left'.")
        if "right" not in data:
            raise KeyError("data must contain the key 'right'.")
        if "parent_id" not in data:
            raise KeyError("data must contain the key 'parent_id'.")
        if "left_child_id" not in data:
            raise KeyError("data must contain the key 'left_child_id'.")
        if "right_child_id" not in data:
            raise KeyError("data must contain the key 'right_child_id'.")
        if "weight" not in data:
            raise KeyError("data must contain the key 'weight'.")
        if "mass" not in data:
            raise KeyError("data must contain the key 'mass'.")
        if "depth" not in data:
            raise KeyError("data must contain the key 'depth'.")
        if "hits_left" not in data:
            raise KeyError("data must contain the key 'hits_left'.")
        # initializations
        result = cls(
            data["left"],
            data["right"],
            data["parent_id"],
            data["left_child_id"],
            data["right_child_id"],
            data["weight"],
            data["mass"],
            data["depth"],
            data["hits_left"],
        )
        return result

    def save(self, path: str | Path, overwrite: bool = False) -> None:
        """
        Save the node to a json file.

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
    def load(cls, path: str | Path) -> TransmitterNode:
        """
        Load the node from a json file.

        Args:
            path (str | Path): The path to the file.

        Returns:
            TransmitterNode: The TransmitterNode instance.
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
        Set an attribute of the node.

        Args:
            name (str): The name of the attribute.
            value (Any): The value of the attribute.
        """
        # freeze check
        if getattr(self, "_frozen", False):
            raise AttributeError(f"{self.__class__.__name__} is immutable")
        # set the attribute
        super().__setattr__(name, value)
        return

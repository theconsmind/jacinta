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
        parent (NDSpace | None): The parent NDSpace of the NDSpace.
        root (NDSpace): The root NDSpace of the NDSpace.
        split_point (NDPoint | None): The split NDPoint of the NDSpace.
        children (tuple[NDSpace, ...] | None): The children NDSpaces of the NDSpace.
        depth (int): The depth of the NDSpace.
        height (int): The height of the NDSpace.
    """

    __slots__ = (
        "_bounds",
        "_parent",
        "_root",
        "_split_point",
        "_children",
        "_depth",
        "_height",
        "_frozen",
    )

    def __init__(
        self,
        bounds: tuple[tuple[float, float], ...],
        parent: NDSpace | None = None,
        split_point: NDPoint | None = None,
    ) -> None:
        """
        Initialize an NDSpace.

        Args:
            bounds (tuple[tuple[float, float], ...]): The bounds of the NDSpace.
            parent (NDSpace | None): The parent NDSpace of the NDSpace.
                Defaults to None.
            split_point (NDPoint | None): The split NDPoint of the NDSpace.
                Defaults to None.
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
        # parent validations
        if parent is not None:
            if not isinstance(parent, NDSpace):
                raise TypeError("parent must be an NDSpace.")
            if parent.nd != len(bounds):
                raise ValueError(f"parent must be {len(bounds)}D.")
            if not all(
                parent_lower <= lower and upper <= parent_upper
                for (lower, upper), (parent_lower, parent_upper) in zip(
                    bounds, parent._bounds, strict=True
                )
            ):
                raise ValueError("bounds must be contained in parent bounds.")
        # split_point validations
        if split_point is not None:
            if not isinstance(split_point, NDPoint):
                raise TypeError("split_point must be an NDPoint.")
            if split_point.nd != len(bounds):
                raise ValueError(f"split_point must be {len(bounds)}D.")
            if not all(
                lower <= coord < upper
                for coord, (lower, upper) in zip(
                    split_point.coordinates, bounds, strict=True
                )
            ):
                raise ValueError("split_point must be contained in bounds.")

        # initializations
        object.__setattr__(self, "_frozen", False)
        self._bounds = tuple((float(lower), float(upper)) for (lower, upper) in bounds)
        self._parent = parent
        self._root = parent.root if parent is not None else self
        self._depth = parent.depth + 1 if parent is not None else 0
        self._split_point = None
        self._children = None
        self._height = 0
        if split_point is not None:
            self.split(split_point)
        object.__setattr__(self, "_frozen", True)
        return

    def __repr__(self) -> str:
        """
        Get the representation of the NDSpace.

        Returns:
            str: The representation of the NDSpace.
        """
        result = f"{self.__class__.__name__}(bounds={self._bounds!r})"
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
    def parent(self) -> NDSpace | None:
        """
        Get the parent NDSpace of the NDSpace.

        Returns:
            NDSpace | None: The parent NDSpace of the NDSpace.
        """
        return self._parent

    @property
    def root(self) -> NDSpace:
        """
        Get the root NDSpace of the NDSpace.

        Returns:
            NDSpace: The root NDSpace of the NDSpace.
        """
        return self._root

    @property
    def split_point(self) -> NDPoint | None:
        """
        Get the split NDPoint of the NDSpace.

        Returns:
            NDPoint | None: The split NDPoint of the NDSpace.
        """
        return self._split_point

    @property
    def children(self) -> tuple[NDSpace, ...] | None:
        """
        Get the children NDSpaces of the NDSpace.

        Returns:
            tuple[NDSpace, ...] | None: The children NDSpaces of the NDSpace.
        """
        return self._children

    @property
    def depth(self) -> int:
        """
        Get the depth of the NDSpace.

        Returns:
            int: The depth of the NDSpace.
        """
        return self._depth

    @property
    def height(self) -> int:
        """
        Get the height of the NDSpace.

        Returns:
            int: The height of the NDSpace.
        """
        return self._height

    @property
    def nd(self) -> int:
        """
        Get the number of dimensions of the NDSpace.

        Returns:
            int: The number of dimensions of the NDSpace.
        """
        nd = len(self._bounds)
        return nd

    @property
    def is_leaf(self) -> bool:
        """
        Check if the NDSpace is a leaf.

        Returns:
            bool: True if the NDSpace is a leaf, False otherwise.
        """
        is_leaf = self._split_point is None
        return is_leaf

    def __eq__(self, other: object) -> bool:
        """
        Check if two NDSpaces are equal.

        Args:
            other (object): The object to compare with.

        Returns:
            bool: True if the NDSpaces are equal, False otherwise.
        """
        # other validations
        if type(self) is not type(other):
            return NotImplemented
        # equality check
        result = self._bounds == other._bounds
        return result

    def __contains__(self, other: object) -> bool:
        """
        Check if an NDPoint or NDSpace is within the bounds of the NDSpace.

        Args:
            other (object): The object to check.

        Returns:
            bool: True if the NDPoint or NDSpace is within the bounds, False otherwise.
        """
        # other validations
        if not isinstance(other, (NDPoint, NDSpace)):
            raise TypeError("other must be an NDPoint or an NDSpace.")
        if other.nd != self.nd:
            raise ValueError(f"other must be {self.nd}D.")
        # check if the NDPoint is within the bounds
        if isinstance(other, NDPoint):
            result = all(
                lower <= coord < upper
                for coord, (lower, upper) in zip(
                    other.coordinates, self._bounds, strict=True
                )
            )
        # check if the NDSpace is within the bounds
        elif isinstance(other, NDSpace):
            result = all(
                lower <= other_lower and other_upper <= upper
                for (other_lower, other_upper), (lower, upper) in zip(
                    other._bounds, self._bounds, strict=True
                )
            )
        else:
            result = False
        return result

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
        if point not in self:
            raise ValueError("point must be contained in self.")
        if self._split_point is not None:
            raise ValueError("NDSpace is already split.")
        # split the NDSpace
        spaces = []
        # generate all combinations of upper/lower halves
        for directions in product((False, True), repeat=self.nd):
            new_bounds = list(self._bounds)
            is_valid = True
            # build bounds for each sub-NDSpace
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
            # create new NDSpace if valid (lower < upper)
            if is_valid:
                space = NDSpace(tuple(new_bounds), parent=self)
                spaces.append(space)
        spaces = tuple(spaces)
        # update children
        object.__setattr__(self, "_frozen", False)
        self._split_point = point
        self._children = spaces
        object.__setattr__(self, "_frozen", True)
        self._update_height()
        return spaces

    def collapse(self) -> None:
        """
        Collapse the NDSpace by removing its children.
        """
        # remove children if the NDSpace is split
        if self._split_point is not None:
            # remove parent references from children
            for child in self._children:
                object.__setattr__(child, "_frozen", False)
                child._parent = None
                object.__setattr__(child, "_frozen", True)
                child._update_root()
                child._update_depth()
            # remove children reference from self
            object.__setattr__(self, "_frozen", False)
            self._split_point = None
            self._children = None
            object.__setattr__(self, "_frozen", True)
            self._update_height()
        return

    def add_dimensions(
        self,
        bounds: tuple[tuple[float, float], ...],
    ) -> NDSpace:
        """
        Add new dimensions to the NDSpace.

        Args:
            bounds (tuple[tuple[float, float], ...]): The bounds of the new dimensions.

        Returns:
            NDSpace: The NDSpace with added dimensions.
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
        # add new bounds to the whole NDSpace tree
        new_bounds = tuple((float(lower), float(upper)) for (lower, upper) in bounds)
        new_coords = tuple(lower for lower, _ in new_bounds)
        root = self.root

        def _add_dimensions(space: NDSpace) -> None:
            """
            Recursively add new dimensions to the NDSpace tree.

            Args:
                space (NDSpace): The NDSpace to add new dimensions to.
            """
            object.__setattr__(space, "_frozen", False)
            space._bounds = space._bounds + new_bounds
            # if there is a split NDPoint, update its coordinates
            if space._split_point is not None:
                space._split_point = NDPoint(
                    space._split_point.coordinates + new_coords
                )
                # add new bounds to children
                for child in space._children:
                    _add_dimensions(child)
            object.__setattr__(space, "_frozen", True)
            return

        _add_dimensions(root)
        return self

    def remove_dimensions(self, dims: set[int]) -> NDSpace:
        """
        Remove dimensions from the NDSpace.

        Args:
            dims (set[int]): The indices of the dimensions to remove.

        Returns:
            NDSpace: The NDSpace with removed dimensions.
        """
        # dims validations
        if not isinstance(dims, (set, tuple, list)):
            raise TypeError("dims must be a set.")
        if len(set(dims)) != len(dims):
            raise ValueError("All dims must be unique.")
        for dim in dims:
            if not isinstance(dim, int):
                raise TypeError("All dims must be ints.")
            if not (0 <= dim < self.nd):
                raise IndexError("All dims must be in range.")
        # remove dimensions from the whole NDSpace tree
        dims = set(dims)
        root = self.root

        def _remove_dimensions(space: NDSpace) -> None:
            """
            Recursively remove dimensions from the NDSpace tree.

            Args:
                space (NDSpace): The NDSpace to remove dimensions from.
            """
            object.__setattr__(space, "_frozen", False)
            new_bounds = tuple(
                bound for idx, bound in enumerate(space._bounds) if idx not in dims
            )
            space._bounds = new_bounds
            # if there is a split NDPoint, update its coordinates
            if space._split_point is not None:
                new_coords = tuple(
                    coord
                    for idx, coord in enumerate(space._split_point.coordinates)
                    if idx not in dims
                )
                space._split_point = NDPoint(new_coords)
                # remove dimensions from children
                for child in space._children:
                    _remove_dimensions(child)
                # group duplicated children
                groups = {}
                for child in space._children:
                    if child._bounds not in groups:
                        groups[child._bounds] = []
                    groups[child._bounds].append(child)
                # merge duplicated children
                merged_children = []
                discarded_children = []
                for children in groups.values():
                    selected_child = space._merge(tuple(children))
                    merged_children.append(selected_child)
                    for child in children:
                        if child is not selected_child:
                            discarded_children.append(child)
                space._children = tuple(merged_children)
                # remove parent references from discarded children
                for child in discarded_children:
                    object.__setattr__(child, "_frozen", False)
                    child._parent = None
                    object.__setattr__(child, "_frozen", True)
                    child._update_root()
                    child._update_depth()
            object.__setattr__(space, "_frozen", True)
            space._update_height()
            return

        _remove_dimensions(root)
        return self

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

        def _to_dict(space: NDSpace) -> dict[str, Any]:
            """
            Recursively convert the NDSpace tree to a dictionary.

            Args:
                space (NDSpace): The NDSpace to convert.

            Returns:
                dict[str, Any]: The dictionary representation of the NDSpace.
            """
            result = {
                "type": space.__class__.__name__,
                "bounds": space._bounds,
                "split_point": space._split_point.to_dict()
                if space._split_point is not None
                else None,
                "children": tuple(_to_dict(child) for child in space._children)
                if space._children is not None
                else None,
            }
            return result

        result = _to_dict(self)
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

        def _from_dict(data: dict[str, Any], parent: NDSpace | None = None) -> NDSpace:
            """
            Recursively convert a dictionary to an NDSpace tree.

            Args:
                data (dict[str, Any]): The dictionary representation of the NDSpace.
                parent (NDSpace | None): The parent NDSpace of the NDSpace.
                    Defaults to None.

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
            if "split_point" not in data:
                raise KeyError("data must contain the key 'split_point'.")
            if "children" not in data:
                raise KeyError("data must contain the key 'children'.")
            if (data["split_point"] is None) != (data["children"] is None):
                raise ValueError(
                    "data['split_point'] and data['children'] must be both None "
                    "or both not None."
                )
            # initializations
            space = cls(data["bounds"], parent)
            if data["split_point"] is not None:
                split_point = NDPoint.from_dict(data["split_point"])
                children = tuple(
                    _from_dict(child_data, space) for child_data in data["children"]
                )
                # validate split integrity
                expected_children = cls(data["bounds"]).split(split_point)
                actual_bounds = {child.bounds for child in children}
                expected_bounds = {child.bounds for child in expected_children}
                if actual_bounds != expected_bounds:
                    raise ValueError("children are not compatible with split_point.")
                object.__setattr__(space, "_frozen", False)
                space._split_point = split_point
                space._children = children
                object.__setattr__(space, "_frozen", True)
                space._update_height()
            return space

        space = _from_dict(data)
        return space

    def save(self, path: str | Path, overwrite: bool = False) -> None:
        """
        Save the NDSpace to a json file.

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

    def _update_root(self) -> None:
        """
        Recursively update the root NDSpace of the NDSpace tree.
        """
        new_root = self._parent._root if self._parent is not None else self
        if self._root is not new_root:
            object.__setattr__(self, "_frozen", False)
            self._root = new_root
            object.__setattr__(self, "_frozen", True)
            if self._split_point is not None:
                for child in self._children:
                    child._update_root()
        return

    def _update_depth(self) -> None:
        """
        Recursively update the depth of the NDSpace and its children.
        """
        new_depth = self._parent._depth + 1 if self._parent is not None else 0
        if self._depth != new_depth:
            object.__setattr__(self, "_frozen", False)
            self._depth = new_depth
            object.__setattr__(self, "_frozen", True)
            if self._split_point is not None:
                for child in self._children:
                    child._update_depth()
        return

    def _update_height(self) -> None:
        """
        Recursively update the height of the NDSpace and its ancestors.
        """
        new_height = (
            1 + max(child._height for child in self._children)
            if self._split_point is not None
            else 0
        )
        if self._height != new_height:
            object.__setattr__(self, "_frozen", False)
            self._height = new_height
            object.__setattr__(self, "_frozen", True)
            if self._parent is not None:
                self._parent._update_height()
        return

    def _merge(self, spaces: tuple[NDSpace, ...]) -> NDSpace:
        """
        Merge duplicated NDSpaces into a single NDSpace.

        Args:
            spaces (tuple[NDSpace, ...]): The NDSpaces to merge.

        Returns:
            NDSpace: The selected NDSpace.
        """
        # choose the space with the greatest height
        space = max(spaces, key=lambda space: space.height)
        return space

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
        object.__setattr__(self, name, value)
        return

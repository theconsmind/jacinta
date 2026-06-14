from __future__ import annotations

import json
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, Self

from .NDPoint import NDPoint


class NDSpace:
    """
    An NDSpace represents an N-dimensional space.

    Attributes:
        bounds (tuple[tuple[float, float], ...]): The bounds of the space.
        parent (Self | None): The parent of the space.
        split_point (NDPoint | None): The split point of the space.
        children (tuple[Self, ...] | None): The children of the space.
        root (Self): The root of the space.
        depth (int): The depth of the space.
        height (int): The height of the space.
        min_width (float | None): The minimum width of each dimension of the space.
        max_depth (int | None): The maximum depth of the space.
    """

    __slots__ = (
        "_bounds",
        "_parent",
        "_split_point",
        "_children",
        "_root",
        "_depth",
        "_height",
        "_min_width",
        "_max_depth",
        "_frozen",
    )

    def __init__(
        self,
        bounds: tuple[tuple[float, float], ...],
        min_width: float | None = None,
        max_depth: int | None = None,
    ) -> None:
        """
        Initialize an NDSpace.

        Args:
            bounds (tuple[tuple[float, float], ...]): The bounds of the space.
            min_width (float | None): The minimum width of each dimension of the space.
                Defaults to None.
            max_depth (int | None): The maximum depth of the space.
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
        # min_width validations
        if min_width is not None:
            if not isinstance(min_width, (float, int)):
                raise TypeError("min_width must be a float.")
            if min_width <= 0:
                raise ValueError("min_width must be greater than 0.")
            if any(upper - lower < min_width for lower, upper in bounds):
                raise ValueError(
                    "All bounds widths must be greater than or equal to min_width."
                )
        # max_depth validations
        if max_depth is not None:
            if not isinstance(max_depth, int):
                raise TypeError("max_depth must be an int.")
            if max_depth < 0:
                raise ValueError("max_depth must be greater than or equal to 0.")
        # initializations
        object.__setattr__(self, "_frozen", False)
        self._bounds = tuple((float(lower), float(upper)) for lower, upper in bounds)
        self._parent = None
        self._split_point = None
        self._children = None
        self._root = self
        self._depth = 0
        self._height = 0
        self._min_width = float(min_width) if min_width is not None else None
        self._max_depth = max_depth
        object.__setattr__(self, "_frozen", True)
        return

    def __repr__(self) -> str:
        """
        Get the representation of the space.

        Returns:
            str: The representation of the space.
        """
        result = f"{self.__class__.__name__}(bounds={self._bounds!r})"
        return result

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        """
        Get the bounds of the space.

        Returns:
            tuple[tuple[float, float], ...]: The bounds of the space.
        """
        return self._bounds

    @property
    def parent(self) -> Self | None:
        """
        Get the parent of the space.

        Returns:
            Self | None: The parent of the space.
        """
        return self._parent

    @property
    def split_point(self) -> NDPoint | None:
        """
        Get the split point of the space.

        Returns:
            NDPoint | None: The split point of the space.
        """
        return self._split_point

    @property
    def children(self) -> tuple[Self, ...] | None:
        """
        Get the children of the space.

        Returns:
            tuple[Self, ...] | None: The children of the space.
        """
        return self._children

    @property
    def root(self) -> Self:
        """
        Get the root of the space.

        Returns:
            Self: The root of the space.
        """
        return self._root

    @property
    def depth(self) -> int:
        """
        Get the depth of the space.

        Returns:
            int: The depth of the space.
        """
        return self._depth

    @property
    def height(self) -> int:
        """
        Get the height of the space.

        Returns:
            int: The height of the space.
        """
        return self._height

    @property
    def min_width(self) -> float | None:
        """
        Get the minimum width of each dimension of the space.

        Returns:
            float | None: The minimum width of each dimension of the space.
        """
        return self._min_width

    @property
    def max_depth(self) -> int | None:
        """
        Get the maximum depth of the space.

        Returns:
            int | None: The maximum depth of the space.
        """
        return self._max_depth

    @property
    def nd(self) -> int:
        """
        Get the number of dimensions of the space.

        Returns:
            int: The number of dimensions of the space.
        """
        nd = len(self._bounds)
        return nd

    @property
    def is_leaf(self) -> bool:
        """
        Check if the space is a leaf.

        Returns:
            bool: True if the space is a leaf, False otherwise.
        """
        is_leaf = self._children is None
        return is_leaf

    def __eq__(self, other: object) -> bool:
        """
        Check if two spaces are equal.

        Args:
            other (object): The object to compare with.

        Returns:
            bool: True if the spaces are equal, False otherwise.
        """
        # other validations
        if type(self) is not type(other):
            return NotImplemented
        # equality check
        result = self._bounds == other._bounds
        return result

    def __contains__(self, other: object) -> bool:
        """
        Check if a point or space is within the bounds of the space.

        Args:
            other (object): The object to check.

        Returns:
            bool: True if the point or space is within the bounds of the space,
                False otherwise.
        """
        # other validations
        if not isinstance(other, (NDPoint, NDSpace)):
            raise TypeError("other must be an NDPoint or an NDSpace.")
        if other.nd != self.nd:
            raise ValueError(f"other must be {self.nd}D.")
        # check if the point is within the bounds
        result = False
        if isinstance(other, NDPoint):
            result = all(
                lower <= coord < upper
                for coord, (lower, upper) in zip(
                    other.coordinates, self._bounds, strict=True
                )
            )
        # check if the space is within the bounds
        elif isinstance(other, NDSpace):
            result = all(
                lower <= other_lower and other_upper <= upper
                for (other_lower, other_upper), (lower, upper) in zip(
                    other._bounds, self._bounds, strict=True
                )
            )
        return result

    def find_leaf(self, point: NDPoint) -> Self:
        """
        Find the leaf that contains the point.

        Args:
            point (NDPoint): The point to find the leaf for.

        Returns:
            Self: The leaf that contains the point.
        """
        # point validations
        if not isinstance(point, NDPoint):
            raise TypeError("point must be an NDPoint.")
        if point.nd != self.nd:
            raise ValueError(f"point must be {self.nd}D.")
        if point not in self:
            raise ValueError("point must be contained in self.")
        # find the leaf that contains the point
        space = self
        while not space.is_leaf:
            for child in space._children:
                if point in child:
                    space = child
                    break
        return space

    def can_split(self, point: NDPoint) -> bool:
        """
        Check if the space can be split by a point.

        Args:
            point (NDPoint): The point to check if the space can be split by.

        Returns:
            bool: True if the space can be split, False otherwise.
        """
        # point validations
        if not isinstance(point, NDPoint):
            raise TypeError("point must be an NDPoint.")
        if point.nd != self.nd:
            raise ValueError(f"point must be {self.nd}D.")
        if point not in self:
            raise ValueError("point must be contained in self.")
        # check if the space is a leaf
        result = True
        if not self.is_leaf:
            result = False
        # check if the space is at max depth
        elif self._max_depth is not None and self._depth == self._max_depth:
            result = False
        # check if the space can be split by the point
        elif self._min_width is not None:
            for coord, (lower, upper) in zip(
                point.coordinates, self._bounds, strict=True
            ):
                lower_width = coord - lower
                upper_width = upper - coord
                # skip new empty bounds (lower == upper)
                if lower_width != 0 and lower_width < self._min_width:
                    result = False
                    break
                if upper_width != 0 and upper_width < self._min_width:
                    result = False
                    break
        return result

    def split(self, point: NDPoint) -> tuple[Self, ...]:
        """
        Split the space into smaller spaces based on a point.

        Args:
            point (NDPoint): The point to split the space by.

        Returns:
            tuple[Self, ...]: The sub-spaces created by the split.
        """
        # point validations
        if not isinstance(point, NDPoint):
            raise TypeError("point must be an NDPoint.")
        if point.nd != self.nd:
            raise ValueError(f"point must be {self.nd}D.")
        if point not in self:
            raise ValueError("point must be contained in self.")
        # self validations
        if not self.can_split(point):
            raise ValueError("self cannot be split.")
        # split the space
        spaces = []
        # generate all combinations of upper/lower halves
        for directions in product((False, True), repeat=self.nd):
            new_bounds = list(self._bounds)
            is_valid = True
            # build bounds for each sub-space
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
            # create new space if valid (lower < upper)
            if is_valid:
                space = self.__class__(
                    tuple(new_bounds), self._min_width, self._max_depth
                )
                object.__setattr__(space, "_frozen", False)
                space._parent = self
                space._root = self._root
                space._depth = self._depth + 1
                object.__setattr__(space, "_frozen", True)
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
        Collapse the space by removing its children.
        """
        # remove children if the space is split
        if not self.is_leaf:
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
    ) -> Self:
        """
        Add new dimensions to the space.

        Args:
            bounds (tuple[tuple[float, float], ...]): The bounds of the new dimensions.

        Returns:
            Self: The space with added dimensions.
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
            if self._min_width is not None and bound[1] - bound[0] < self._min_width:
                raise ValueError(
                    "All bounds widths must be greater than or equal to min_width."
                )
        # add new bounds to the whole tree
        new_bounds = tuple((float(lower), float(upper)) for lower, upper in bounds)
        new_coords = tuple(lower for lower, _ in new_bounds)
        root = self.root

        def _add_dimensions(space: Self) -> None:
            """
            Recursively add new dimensions to the tree.

            Args:
                space (Self): The space to add new dimensions to.
            """
            object.__setattr__(space, "_frozen", False)
            space._bounds = space._bounds + new_bounds
            # add new bounds to children
            if not space.is_leaf:
                space._split_point = NDPoint(
                    space._split_point.coordinates + new_coords
                )
                for child in space._children:
                    _add_dimensions(child)
            object.__setattr__(space, "_frozen", True)
            return

        _add_dimensions(root)
        return self

    def remove_dimensions(self, dims: set[int]) -> Self:
        """
        Remove dimensions from the space.

        Args:
            dims (set[int]): The indices of the dimensions to remove.

        Returns:
            Self: The space with removed dimensions.
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
        # remove dimensions from the whole tree
        dims = set(dims)
        root = self.root

        def _remove_dimensions(space: Self) -> None:
            """
            Recursively remove dimensions from the tree.

            Args:
                space (Self): The space to remove dimensions from.
            """
            object.__setattr__(space, "_frozen", False)
            new_bounds = tuple(
                bound for idx, bound in enumerate(space._bounds) if idx not in dims
            )
            space._bounds = new_bounds
            # remove dimensions from children
            if not space.is_leaf:
                new_coords = tuple(
                    coord
                    for idx, coord in enumerate(space._split_point.coordinates)
                    if idx not in dims
                )
                space._split_point = NDPoint(new_coords)
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

    def copy(self) -> Self:
        """
        Get a copy of the space.

        Returns:
            Self: The copy of the space.
        """
        result = deepcopy(self)
        return result

    def to_dict(self) -> dict[str, Any]:
        """
        Get the dictionary representation of the space.

        Returns:
            dict[str, Any]: The dictionary representation of the space.
        """

        def _to_dict(space: Self) -> dict[str, Any]:
            """
            Recursively convert the tree to a dictionary.

            Args:
                space (Self): The space to convert.

            Returns:
                dict[str, Any]: The dictionary representation of the space.
            """
            result = {
                "type": space.__class__.__name__,
                "bounds": space._bounds,
                "min_width": space._min_width,
                "max_depth": space._max_depth,
                "split_point": (
                    space._split_point.to_dict()
                    if space._split_point is not None
                    else None
                ),
                "children": (
                    tuple(_to_dict(child) for child in space._children)
                    if not space.is_leaf
                    else None
                ),
            }
            return result

        result = _to_dict(self)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """
        Create a space from a dictionary.

        Args:
            data (dict[str, Any]): The dictionary representation of the space.

        Returns:
            Self: The space.
        """

        def _from_dict(data: dict[str, Any], parent: Self | None = None) -> Self:
            """
            Recursively convert a dictionary to a tree.

            Args:
                data (dict[str, Any]): The dictionary representation of the space.
                parent (Self | None): The parent of the space.
                    Defaults to None.

            Returns:
                Self: The space.
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
            if "min_width" not in data:
                raise KeyError("data must contain the key 'min_width'.")
            if "max_depth" not in data:
                raise KeyError("data must contain the key 'max_depth'.")
            if "split_point" not in data:
                raise KeyError("data must contain the key 'split_point'.")
            if "children" not in data:
                raise KeyError("data must contain the key 'children'.")
            if (data["split_point"] is None) != (data["children"] is None):
                raise ValueError(
                    "data['split_point'] and data['children'] must be both None "
                    "or both not None."
                )
            # parent validations
            if parent is not None:
                if parent._max_depth is not None and parent._depth == parent._max_depth:
                    raise ValueError("parent cannot be split.")
                if parent._min_width != data["min_width"]:
                    raise ValueError(
                        "data['min_width'] must be equal to parent._min_width."
                    )
                if parent._max_depth != data["max_depth"]:
                    raise ValueError(
                        "data['max_depth'] must be equal to parent._max_depth."
                    )
            # initializations
            space = cls(data["bounds"], data["min_width"], data["max_depth"])
            # update parent attributes
            if parent is not None:
                object.__setattr__(space, "_frozen", False)
                space._parent = parent
                space._root = parent._root
                space._depth = parent._depth + 1
                object.__setattr__(space, "_frozen", True)
            # update children attributes
            if data["children"] is not None:
                split_point = NDPoint.from_dict(data["split_point"])
                children = tuple(
                    _from_dict(child_data, space) for child_data in data["children"]
                )
                # validate split integrity
                expected_children = cls(
                    data["bounds"], data["min_width"], data["max_depth"]
                ).split(split_point)
                actual_bounds = {child._bounds for child in children}
                expected_bounds = {child._bounds for child in expected_children}
                if (
                    len(children) != len(expected_children)
                    or actual_bounds != expected_bounds
                ):
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
        Save the space to a json file.

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
        Load a space from a json file.

        Args:
            path (str | Path): The path to the file.

        Returns:
            Self: The space.
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
        Recursively update the root of the tree.
        """
        new_root = self._parent._root if self._parent is not None else self
        if self._root is not new_root:
            object.__setattr__(self, "_frozen", False)
            self._root = new_root
            object.__setattr__(self, "_frozen", True)
            if not self.is_leaf:
                for child in self._children:
                    child._update_root()
        return

    def _update_depth(self) -> None:
        """
        Recursively update the depth of the space and its children.
        """
        new_depth = self._parent._depth + 1 if self._parent is not None else 0
        if self._depth != new_depth:
            object.__setattr__(self, "_frozen", False)
            self._depth = new_depth
            object.__setattr__(self, "_frozen", True)
            if not self.is_leaf:
                for child in self._children:
                    child._update_depth()
        return

    def _update_height(self) -> None:
        """
        Recursively update the height of the space and its ancestors.
        """
        new_height = (
            1 + max(child._height for child in self._children)
            if not self.is_leaf
            else 0
        )
        if self._height != new_height:
            object.__setattr__(self, "_frozen", False)
            self._height = new_height
            object.__setattr__(self, "_frozen", True)
            if self._parent is not None:
                self._parent._update_height()
        return

    def _merge(self, spaces: tuple[Self, ...]) -> Self:
        """
        Merge duplicated spaces into a single space.

        Args:
            spaces (tuple[Self, ...]): The spaces to merge.

        Returns:
            Self: The selected space.
        """
        # choose the space with the greatest height
        space = max(spaces, key=lambda space: space.height)
        return space

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Set an attribute of the space.

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

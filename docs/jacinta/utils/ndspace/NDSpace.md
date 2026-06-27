# NDSpace

## Overview

[`NDSpace`](../../../../src/jacinta/utils/ndspace/NDSpace.py) is an N-dimensional region of continuous space that forms the structural backbone of Jacinta's adaptive trees. It recursively partitions space into progressively smaller subspaces, allowing different regions to be represented at different levels of precision.

While `NDSpace` provides the common representation of a spatial region, specialized subclasses can extend it with additional properties and capabilities.

## API Reference

```python
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
```

### Constructor

```python
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
```

### Properties

```python
@property
def bounds(self) -> tuple[tuple[float, float], ...]:
    """
    Get the bounds of the space.

    Returns:
        tuple[tuple[float, float], ...]: The bounds of the space.
    """

@property
def parent(self) -> Self | None:
    """
    Get the parent of the space.

    Returns:
        Self | None: The parent of the space.
    """

@property
def split_point(self) -> NDPoint | None:
    """
    Get the split point of the space.

    Returns:
        NDPoint | None: The split point of the space.
    """

@property
def children(self) -> tuple[Self, ...] | None:
    """
    Get the children of the space.

    Returns:
        tuple[Self, ...] | None: The children of the space.
    """

@property
def root(self) -> Self:
    """
    Get the root of the space.

    Returns:
        Self: The root of the space.
    """

@property
def depth(self) -> int:
    """
    Get the depth of the space.

    Returns:
        int: The depth of the space.
    """

@property
def height(self) -> int:
    """
    Get the height of the space.

    Returns:
        int: The height of the space.
    """

@property
def min_width(self) -> float | None:
    """
    Get the minimum width of each dimension of the space.

    Returns:
        float | None: The minimum width of each dimension of the space.
    """

@property
def max_depth(self) -> int | None:
    """
    Get the maximum depth of the space.

    Returns:
        int | None: The maximum depth of the space.
    """

@property
def nd(self) -> int:
    """
    Get the number of dimensions of the space.

    Returns:
        int: The number of dimensions of the space.
    """

@property
def is_leaf(self) -> bool:
    """
    Check if the space is a leaf.

    Returns:
        bool: True if the space is a leaf, False otherwise.
    """
```

### `__eq__(other)`

```python
def __eq__(self, other: object) -> bool:
    """
    Check if two spaces are equal.

    Args:
        other (object): The object to compare with.

    Returns:
        bool: True if the spaces are equal, False otherwise.
    """
```

### `__contains__(other)`

```python
def __contains__(self, other: object) -> bool:
    """
    Check if a point or space is within the bounds of the space.

    Args:
        other (object): The object to check.

    Returns:
        bool: True if the point or space is within the bounds of the space,
            False otherwise.
    """
```

### `find_leaf(point)`

```python
def find_leaf(self, point: NDPoint) -> Self:
    """
    Find the leaf that contains the point.

    Args:
        point (NDPoint): The point to find the leaf for.

    Returns:
        Self: The leaf that contains the point.
    """
```

### `can_split(point)`

```python
def can_split(self, point: NDPoint) -> bool:
    """
    Check if the space can be split by a point.

    Args:
        point (NDPoint): The point to check if the space can be split by.

    Returns:
        bool: True if the space can be split, False otherwise.
    """
```

### `split(point)`

```python
def split(self, point: NDPoint) -> tuple[Self, ...]:
    """
    Split the space into smaller spaces based on a point.

    Args:
        point (NDPoint): The point to split the space by.

    Returns:
        tuple[Self, ...]: The sub-spaces created by the split.
    """
```

### `collapse()`

```python
def collapse(self) -> None:
    """
    Collapse the space by removing its children.
    """
```

### `add_dimensions(bounds)`

```python
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
```

### `remove_dimensions(dims)`

```python
def remove_dimensions(self, dims: set[int]) -> Self:
    """
    Remove dimensions from the space.

    Args:
        dims (set[int]): The indices of the dimensions to remove.

    Returns:
        Self: The space with removed dimensions.
    """
```

### `copy()`

```python
def copy(self) -> Self:
    """
    Get a copy of the space.

    Returns:
        Self: The copy of the space.
    """
```

### `to_dict()`

```python
def to_dict(self) -> dict[str, Any]:
    """
    Get the dictionary representation of the space.

    Returns:
        dict[str, Any]: The dictionary representation of the space.
    """
```

### `from_dict(data)` *(classmethod)*

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> Self:
    """
    Create a space from a dictionary.

    Args:
        data (dict[str, Any]): The dictionary representation of the space.

    Returns:
        Self: The space.
    """
```

### `save(path, overwrite=False)`

```python
def save(self, path: str | Path, overwrite: bool = False) -> None:
    """
    Save the space to a json file.

    Args:
        path (str | Path): The path to the file.
        overwrite (bool): Whether to overwrite the file if it exists.
            Defaults to False.
    """
```

### `load(path)` *(classmethod)*

```python
@classmethod
def load(cls, path: str | Path) -> Self:
    """
    Load a space from a json file.

    Args:
        path (str | Path): The path to the file.

    Returns:
        Self: The space.
    """
```

## Examples

```python
from jacinta.utils.ndspace import NDSpace, NDPoint

# Initialize a 2D NDSpace
space = NDSpace(
    bounds=((0.0, 10.0), (0.0, 10.0)),
)
print(space.nd)       # 2
print(space.is_leaf)  # True
print(space.height)   # 0
print(space.depth)    # 0

# Containment
print(NDPoint((5.0, 5.0)) in space)   # True
print(NDPoint((10.0, 5.0)) in space)  # False
print(NDPoint((20.0, 5.0)) in space)  # False

# Split at the midpoint
midpoint = NDPoint((5.0, 5.0))
children = space.split(midpoint)
print(len(children))            # 4
print(space.is_leaf)            # False
print(space.height)             # 1
print(space.children[0].depth)  # 1

# Find leaf
leaf = space.find_leaf(NDPoint((2.5, 0.5)))
print(leaf.bounds)  # ((0.0, 5.0), (0.0, 5.0))

# Add dimension
space.add_dimensions(((0.0, 5.0),))
print(space.nd)      # 3
print(space.bounds)  # ((0.0, 10.0), (0.0, 10.0), (0.0, 5.0))

# Remove dimension
space.remove_dimensions({0})
print(space.nd)      # 2
print(space.bounds)  # ((0.0, 10.0), (0.0, 5.0))

# Collapse
space.collapse()
print(space.is_leaf)  # True

# Serialize and deserialize
data = space.to_dict()
space2 = NDSpace.from_dict(data)
assert space == space2

# Save and load
space.save("space.json")
space3 = NDSpace.load("space.json")
assert space == space3
```

## Limitations

- `add_dimensions` and `remove_dimensions` operate on the entire tree starting from `self.root`, regardless of which node the method is called on.
- `remove_dimensions` merges siblings with identical regions by keeping the subtree with the greatest `height`. Any discarded subtrees are detached from the original tree and lose their parent references. Existing references to these subtrees will continue to point to the detached trees.
- Point containment uses a half-open interval. A point exactly on the upper boundary of a dimension is not considered part of the region.

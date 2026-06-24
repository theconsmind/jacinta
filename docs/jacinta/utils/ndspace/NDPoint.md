# NDPoint

## Overview

[`NDPoint`](../../../src/jacinta/utils/ndspace/NDPoint.py) is a point in an [`NDSpace`](NDSpace.md). It serves as the fundamental coordinate type across Jacinta, representing any position in a continuous space, whether it corresponds to an observation, an action, or any other spatial value.

While `NDPoint` provides the common coordinate representation, specialized subclasses can extend it with additional properties and capabilities.

## API Reference

```python
class NDPoint:
    """
    An NDPoint represents an N-dimensional point.

    Attributes:
        coordinates (tuple[float, ...]): The coordinates of the point.
    """
```

### Constructor

```python
def __init__(self, coordinates: tuple[float, ...]) -> None:
    """
    Initialize an NDPoint.

    Args:
        coordinates (tuple[float, ...]): The coordinates of the point.
    """
```

### Properties

```python
@property
def coordinates(self) -> tuple[float, ...]:
    """
    Get the coordinates of the point.

    Returns:
        tuple[float, ...]: The coordinates of the point.
    """

@property
def nd(self) -> int:
    """
    Get the number of dimensions of the point.

    Returns:
        int: The number of dimensions of the point.
    """
```

### `__eq__(other)`

```python
def __eq__(self, other: object) -> bool:
    """
    Check if two points are equal.

    Args:
        other (object): The object to compare with.

    Returns:
        bool: True if the points are equal, False otherwise.
    """
```

### `copy()`

```python
def copy(self) -> Self:
    """
    Get a copy of the point.

    Returns:
        Self: The copy of the point.
    """
```

### `to_dict()`

```python
def to_dict(self) -> dict[str, Any]:
    """
    Get the dictionary representation of the point.

    Returns:
        dict[str, Any]: The dictionary representation of the point.
    """
```

### `from_dict(data)` *(classmethod)*

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> Self:
    """
    Create a point from a dictionary.

    Args:
        data (dict[str, Any]): The dictionary representation of the point.

    Returns:
        Self: The point.
    """
```

### `save(path, overwrite=False)`

```python
def save(self, path: str | Path, overwrite: bool = False) -> None:
    """
    Save the point to a json file.

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
    Load a point from a json file.

    Args:
        path (str | Path): The path to the file.

    Returns:
        Self: The point.
    """
```

## Examples

```python
from jacinta.utils.ndspace import NDPoint

# Initialize a 3D NDPoint
point = NDPoint((0.1, 0.5, 0.9))
print(point.nd)           # 3
print(point.coordinates)  # (0.1, 0.5, 0.9)

# Serialize and deserialize
data = point.to_dict()
point2 = NDPoint.from_dict(data)
assert point == point2

# Save and load
point.save("point.json")
point3 = NDPoint.load("point.json")
assert point == point3
```

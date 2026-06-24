# Scheduler

## Overview

[`Scheduler`](../../../../src/jacinta/utils/scheduler/Scheduler.py) is an abstract component that maps each depth level of an [`NDSpace`](../ndspace/NDSpace.md) tree to a floating-point value. It provides a depth-dependent mechanism for configuring parameters that may vary according to the precision requirements of different regions of the space.

While `Scheduler` defines the common depth-to-value mapping interface, specialized subclasses can implement different scheduling strategies to control how parameter values evolve across tree depths.

## API Reference

```python
class Scheduler(ABC):
    """
    A Scheduler represents a strategy that assigns a value to a depth.
    """
```

### `__call__(depth)` *(abstractmethod)*

```python
@abstractmethod
def __call__(self, depth: int) -> float:
    """
    Get the value assigned to the given depth.

    Args:
        depth (int): The depth.

    Returns:
        float: The value assigned to the given depth.
    """
```

### `__eq__(other)` *(abstractmethod)*

```python
@abstractmethod
def __eq__(self, other: object) -> bool:
    """
    Check if two schedulers are equal.

    Args:
        other (object): The object to compare with.

    Returns:
        bool: True if the schedulers are equal, False otherwise.
    """
```

### `copy()`

```python
def copy(self) -> Self:
    """
    Get a copy of the scheduler.

    Returns:
        Self: The copy of the scheduler.
    """
```

### `to_dict()` *(abstractmethod)*

```python
@abstractmethod
def to_dict(self) -> dict[str, Any]:
    """
    Get the dictionary representation of the scheduler.

    Returns:
        dict[str, Any]: The dictionary representation of the scheduler.
    """
```

### `from_dict(data)` *(classmethod, abstractmethod)*

```python
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
```

### `save(path, overwrite=False)`

```python
def save(self, path: str | Path, overwrite: bool = False) -> None:
    """
    Save the scheduler to a json file.

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
    Load a scheduler from a json file.

    Args:
        path (str | Path): The path to the file.

    Returns:
        Self: The scheduler.
    """
```

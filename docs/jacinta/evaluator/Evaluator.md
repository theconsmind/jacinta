# Evaluator

## Overview

[`Evaluator`](../../../src/jacinta/evaluator/Evaluator.py) is an abstract component that transforms raw feedback into a normalized evaluation signal. It provides a common interface for defining feedback normalization strategies that can be used throughout Jacinta.

While `Evaluator` defines the common feedback normalization interface, specialized subclasses can implement different evaluation strategies to produce normalized signals from observed feedback.

## API Reference

```python
class Evaluator(ABC):
    """
    An Evaluator represents a strategy that assigns an advantage to a feedback.
    """
```

### `__call__(feedback)` *(abstractmethod)*

```python
@abstractmethod
def __call__(self, feedback: float) -> float | None:
    """
    Get the advantage produced by the given feedback.

    Args:
        feedback (float): The feedback.

    Returns:
        float | None: The advantage produced by the given feedback.
    """
```

### `__eq__(other)` *(abstractmethod)*

```python
@abstractmethod
def __eq__(self, other: object) -> bool:
    """
    Check if two evaluators are equal.

    Args:
        other (object): The object to compare with.

    Returns:
        bool: True if the evaluators are equal, False otherwise.
    """
```

### `copy()`

```python
def copy(self) -> Self:
    """
    Get a copy of the evaluator.

    Returns:
        Self: The copy of the evaluator.
    """
```

### `to_dict()` *(abstractmethod)*

```python
@abstractmethod
def to_dict(self) -> dict[str, Any]:
    """
    Get the dictionary representation of the evaluator.

    Returns:
        dict[str, Any]: The dictionary representation of the evaluator.
    """
```

### `from_dict(data)` *(classmethod, abstractmethod)*

```python
@classmethod
@abstractmethod
def from_dict(cls, data: dict[str, Any]) -> Self:
    """
    Create an evaluator from a dictionary.

    Args:
        data (dict[str, Any]): The dictionary representation of the evaluator.

    Returns:
        Self: The evaluator.
    """
```

### `save(path, overwrite=False)`

```python
def save(self, path: str | Path, overwrite: bool = False) -> None:
    """
    Save the evaluator to a json file.

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
    Load an evaluator from a json file.

    Args:
        path (str | Path): The path to the file.

    Returns:
        Self: The evaluator.
    """
```

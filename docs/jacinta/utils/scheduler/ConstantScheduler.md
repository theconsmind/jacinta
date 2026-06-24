# ConstantScheduler

## Overview

[`ConstantScheduler`](../../../../src/jacinta/utils/scheduler/ConstantScheduler.py) is a [`Scheduler`](Scheduler.md) that maps every depth level of an [`NDSpace`](../ndspace/NDSpace.md) tree to the same floating-point value. It provides a depth-independent configuration mechanism for parameters that should remain constant across all precision levels.

## Formula

$$
f(\text{depth}) = \text{value}, \quad \text{depth} \in \mathbb{N}_0
$$

## API Reference

```python
class ConstantScheduler(Scheduler):
    """
    A Scheduler that assigns a constant value to a depth.

    Attributes:
        value (float): The constant value of the scheduler.
    """
```

### Constructor

```python
def __init__(self, value: float) -> None:
    """
    Initialize a ConstantScheduler.

    Args:
        value (float): The value of the scheduler.
    """
```

### `__call__(depth)`

```python
def __call__(self, depth: int) -> float:
    """
    Get the value assigned to the given depth.

    Args:
        depth (int): The depth.

    Returns:
        float: The value assigned to the given depth.
    """
```

### Properties

```python
@property
def value(self) -> float:
    """
    Get the value of the scheduler.

    Returns:
        float: The value of the scheduler.
    """
```

### `__eq__(other)`

```python
def __eq__(self, other: object) -> bool:
    """
    Check if two schedulers are equal.

    Args:
        other (object): The object to compare with.

    Returns:
        bool: True if the schedulers are equal, False otherwise.
    """
```

### `to_dict()`

```python
def to_dict(self) -> dict[str, Any]:
    """
    Get the dictionary representation of the scheduler.

    Returns:
        dict[str, Any]: The dictionary representation of the scheduler.
    """
```

### `from_dict(data)` *(classmethod)*

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> ConstantScheduler:
    """
    Create a scheduler from a dictionary.

    Args:
        data (dict[str, Any]): The dictionary representation of the scheduler.

    Returns:
        ConstantScheduler: The scheduler.
    """
```

### Inherited API

`ConstantScheduler` inherits from [`Scheduler`](Scheduler.md).

## Examples

```python
from jacinta.utils.scheduler import ConstantScheduler

# Initialize a ConstantScheduler
scheduler = ConstantScheduler(
    value=10.0,
)
print(scheduler(0))   # 10.0
print(scheduler(3))   # 10.0
print(scheduler(10))  # 10.0

# Serialize and deserialize
data = scheduler.to_dict()
scheduler2 = ConstantScheduler.from_dict(data)
assert scheduler == scheduler2

# Save and load
scheduler.save("scheduler.json")
scheduler3 = ConstantScheduler.load("scheduler.json")
assert scheduler == scheduler3
```

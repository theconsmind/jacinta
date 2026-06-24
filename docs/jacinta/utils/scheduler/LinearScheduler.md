# LinearScheduler

## Overview

[`LinearScheduler`](../../../src/jacinta/utils/scheduler/LinearScheduler.py) is a [`Scheduler`](Scheduler.md) that maps each depth level of an [`NDSpace`](../ndspace/NDSpace.md) tree to a floating-point value according to a linear function. It provides a depth-dependent configuration mechanism for parameters that must evolve linearly across different precision levels.

## Formula

$$
f(\text{depth}) = \text{slope} \cdot \text{depth} + \text{intercept}, \quad \text{depth} \in \mathbb{N}_0
$$

## API Reference

```python
class LinearScheduler(Scheduler):
    """
    A Scheduler that assigns a linear value to a depth.

    Attributes:
        slope (float): The slope of the linear function.
        intercept (float): The intercept of the linear function.
        min_value (float | None): The minimum value of the scheduler.
        max_value (float | None): The maximum value of the scheduler.
    """
```

### Constructor

```python
def __init__(
    self,
    slope: float,
    intercept: float,
    min_value: float | None = None,
    max_value: float | None = None,
) -> None:
    """
    Initialize a LinearScheduler.

    Args:
        slope (float): The slope of the linear function.
        intercept (float): The intercept of the linear function.
        min_value (float | None): The minimum value of the scheduler.
            Defaults to None.
        max_value (float | None): The maximum value of the scheduler.
            Defaults to None.
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
def slope(self) -> float:
    """
    Get the slope of the scheduler.

    Returns:
        float: The slope of the scheduler.
    """

@property
def intercept(self) -> float:
    """
    Get the intercept of the scheduler.

    Returns:
        float: The intercept of the scheduler.
    """

@property
def min_value(self) -> float | None:
    """
    Get the minimum value of the scheduler.

    Returns:
        float | None: The minimum value of the scheduler.
    """

@property
def max_value(self) -> float | None:
    """
    Get the maximum value of the scheduler.

    Returns:
        float | None: The maximum value of the scheduler.
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
def from_dict(cls, data: dict[str, Any]) -> LinearScheduler:
    """
    Create a scheduler from a dictionary.

    Args:
        data (dict[str, Any]): The dictionary representation of the scheduler.

    Returns:
        LinearScheduler: The scheduler.
    """
```

### Inherited API

`LinearScheduler` inherits from [`Scheduler`](Scheduler.md).

## Examples

```python
from jacinta.utils.scheduler import LinearScheduler

# Initialize a LinearScheduler
scheduler = LinearScheduler(
    slope=1.0,
    intercept=0.0,
    min_value=0.2,
    max_value=10.0,
)
print(scheduler(0))   # 0.2
print(scheduler(3))   # 3.0
print(scheduler(10))  # 10.0

# Serialize and deserialize
data = scheduler.to_dict()
scheduler2 = LinearScheduler.from_dict(data)
assert scheduler == scheduler2

# Save and load
scheduler.save("scheduler.json")
scheduler3 = LinearScheduler.load("scheduler.json")
assert scheduler == scheduler3
```

# ExponentialScheduler

## Overview

[`ExponentialScheduler`](../../../../src/jacinta/utils/scheduler/exponential_scheduler.py) is a [`Scheduler`](scheduler.md) that maps each depth level of an [`NDSpace`](../ndspace/nd_space.md) tree to a floating-point value according to an exponential function. It provides a depth-dependent configuration mechanism for parameters that must evolve exponentially across different precision levels.

## Formula

$$
f(\text{depth}) = \text{scale} \cdot e^{\text{rate} \cdot \text{depth}} + \text{intercept}, \quad \text{depth} \in \mathbb{N}_0
$$

## API Reference

```python
class ExponentialScheduler(Scheduler):
    """
    A Scheduler that assigns an exponential value to a depth.

    Attributes:
        scale (float): The scale of the exponential function.
        rate (float): The rate of the exponential function.
        intercept (float): The intercept of the exponential function.
        min_value (float | None): The minimum value of the scheduler.
        max_value (float | None): The maximum value of the scheduler.
    """
```

### Constructor

```python
def __init__(
    self,
    scale: float,
    rate: float,
    intercept: float,
    min_value: float | None = None,
    max_value: float | None = None,
) -> None:
    """
    Initialize an ExponentialScheduler.

    Args:
        scale (float): The scale of the exponential function.
        rate (float): The rate of the exponential function.
        intercept (float): The intercept of the exponential function.
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
def scale(self) -> float:
    """
    Get the scale of the scheduler.

    Returns:
        float: The scale of the scheduler.
    """

@property
def rate(self) -> float:
    """
    Get the rate of the scheduler.

    Returns:
        float: The rate of the scheduler.
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
def from_dict(cls, data: dict[str, Any]) -> ExponentialScheduler:
    """
    Create a scheduler from a dictionary.

    Args:
        data (dict[str, Any]): The dictionary representation of the scheduler.

    Returns:
        ExponentialScheduler: The scheduler.
    """
```

### Inherited API

`ExponentialScheduler` inherits from [`Scheduler`](scheduler.md).

## Examples

```python
from jacinta.utils.scheduler import ExponentialScheduler

# initialize an ExponentialScheduler
scheduler = ExponentialScheduler(
    scale=0.1,
    rate=1.0,
    intercept=0.0,
    min_value=0.2,
    max_value=10.0,
)
print(scheduler(depth=0))   # 0.2
print(scheduler(depth=3))   # 2.008553692318767
print(scheduler(depth=10))  # 10.0

# serialize and deserialize
data = scheduler.to_dict()
scheduler2 = ExponentialScheduler.from_dict(data)
assert scheduler == scheduler2

# save and load
scheduler.save(path="scheduler.json")
scheduler3 = ExponentialScheduler.load(path="scheduler.json")
assert scheduler == scheduler3
```

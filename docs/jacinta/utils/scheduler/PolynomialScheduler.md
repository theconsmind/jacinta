# PolynomialScheduler

## Overview

[`PolynomialScheduler`](../../../../src/jacinta/utils/scheduler/PolynomialScheduler.py) is a [`Scheduler`](Scheduler.md) that maps each depth level of an [`NDSpace`](../ndspace/NDSpace.md) tree to a floating-point value according to a polynomial function. It provides a depth-dependent configuration mechanism for parameters that must evolve polynomially across different precision levels.

## Formula

$$
f(\text{depth}) = \sum_{i=0}^{n} c_i \cdot \text{depth}^i, \quad \text{depth} \in \mathbb{N}_0
$$

## API Reference

```python
class PolynomialScheduler(Scheduler):
    """
    A Scheduler that assigns a polynomial value to a depth.

    Attributes:
        coefficients (tuple[float, ...]): The coefficients of the polynomial function.
        min_value (float | None): The minimum value of the scheduler.
        max_value (float | None): The maximum value of the scheduler.
    """
```

### Constructor

```python
def __init__(
    self,
    coefficients: tuple[float, ...],
    min_value: float | None = None,
    max_value: float | None = None,
) -> None:
    """
    Initialize a PolynomialScheduler.

    Args:
        coefficients (tuple[float, ...]): The coefficients of the polynomial
            function.
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
def coefficients(self) -> tuple[float, ...]:
    """
    Get the coefficients of the scheduler.

    Returns:
        tuple[float, ...]: The coefficients of the scheduler.
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
def from_dict(cls, data: dict[str, Any]) -> PolynomialScheduler:
    """
    Create a scheduler from a dictionary.

    Args:
        data (dict[str, Any]): The dictionary representation of the scheduler.

    Returns:
        PolynomialScheduler: The scheduler.
    """
```

### Inherited API

`PolynomialScheduler` inherits from [`Scheduler`](Scheduler.md).

## Examples

```python
from jacinta.utils.scheduler import PolynomialScheduler

# Initialize a PolynomialScheduler
scheduler = PolynomialScheduler(
    coefficients=(0.0, 1.0, 0.5),
    min_value=0.2,
    max_value=10.0,
)
print(scheduler(0))   # 0.2
print(scheduler(3))   # 7.5
print(scheduler(10))  # 10.0

# Serialize and deserialize
data = scheduler.to_dict()
scheduler2 = PolynomialScheduler.from_dict(data)
assert scheduler == scheduler2

# Save and load
scheduler.save("scheduler.json")
scheduler3 = PolynomialScheduler.load("scheduler.json")
assert scheduler == scheduler3
```

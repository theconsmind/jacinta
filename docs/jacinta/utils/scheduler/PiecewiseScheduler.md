# PiecewiseScheduler

## Overview

[`PiecewiseScheduler`](../../../../src/jacinta/utils/scheduler/PiecewiseScheduler.py) is a [`Scheduler`](Scheduler.md) that maps each depth level of an [`NDSpace`](../ndspace/NDSpace.md) tree to a floating-point value according to a piecewise function. It provides a depth-dependent configuration mechanism for parameters that must follow different scheduling strategies across different precision levels.

## Formula

$$
f(\text{depth}) =
\begin{cases}
f_0(\text{depth}), & d_0 \leq \text{depth} < d_1 \\
f_1(\text{depth}), & d_1 \leq \text{depth} < d_2 \\
\vdots \\
f_n(\text{depth}), & d_n \leq \text{depth}
\end{cases},
\quad \text{depth} \in \mathbb{N}_0
$$

## API Reference

```python
class PiecewiseScheduler(Scheduler):
    """
    A Scheduler that uses different Schedulers for different depth ranges.

    Attributes:
        segments (tuple[tuple[int, Scheduler], ...]): The segments of the scheduler.
    """
```

### Constructor

```python
def __init__(
    self,
    segments: tuple[tuple[int, Scheduler], ...],
) -> None:
    """
    Initialize a PiecewiseScheduler.

    Args:
        segments (tuple[tuple[int, Scheduler], ...]): The segments of the scheduler.
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
def segments(self) -> tuple[tuple[int, Scheduler], ...]:
    """
    Get the segments of the scheduler.

    Returns:
        tuple[tuple[int, Scheduler], ...]: The segments of the scheduler.
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
def from_dict(cls, data: dict[str, Any]) -> PiecewiseScheduler:
    """
    Create a scheduler from a dictionary.

    Args:
        data (dict[str, Any]): The dictionary representation of the scheduler.

    Returns:
        PiecewiseScheduler: The scheduler.
    """
```

### Inherited API

`PiecewiseScheduler` inherits from [`Scheduler`](Scheduler.md).

## Examples

```python
from jacinta.utils.scheduler import ConstantScheduler, LinearScheduler, PiecewiseScheduler

# Initialize a PiecewiseScheduler
scheduler = PiecewiseScheduler(
    segments=(
        (
            0,
            LinearScheduler(
                slope=1.0,
                intercept=0.0,
            ),
        ),
        (
            5,
            ConstantScheduler(
                value=10.0,
            )
        ),
    ),
)
print(scheduler(0))   # 0.0
print(scheduler(3))   # 3.0
print(scheduler(10))  # 10.0

# Serialize and deserialize
data = scheduler.to_dict()
scheduler2 = PiecewiseScheduler.from_dict(data)
assert scheduler == scheduler2

# Save and load
scheduler.save("scheduler.json")
scheduler3 = PiecewiseScheduler.load("scheduler.json")
assert scheduler == scheduler3
```

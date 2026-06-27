# ZScoreEvaluator

## Overview

[`ZScoreEvaluator`](../../../src/jacinta/evaluator/ZScoreEvaluator.py) is an [`Evaluator`](Evaluator.md) that normalizes raw feedback according to its z-score using online estimates of the mean and variance.

It provides a normalized evaluation strategy in which positive values indicate better-than-average feedback, while negative values indicate worse-than-average feedback.

## Formula

$$
z = \frac{\text{feedback} - \mu}{\sqrt{\sigma^2} + \varepsilon}
$$

## API Reference

```python
class ZScoreEvaluator(Evaluator):
    """
    An Evaluator that assigns a z-score advantage to a feedback.

    Attributes:
        mean (float | None): The mean of the feedback.
        var (float | None): The variance of the feedback.
        mean_ema_rate (float): The mean EMA rate.
        var_ema_rate (float): The variance EMA rate.
        eps (float): A small positive value used for numerical stability.
    """
```

### Constructor

```python
def __init__(
    self,
    mean_ema_rate: float,
    var_ema_rate: float,
    eps: float = 1e-9,
) -> None:
    """
    Initialize a ZScoreEvaluator.

    Args:
        mean_ema_rate (float): The mean EMA rate.
        var_ema_rate (float): The variance EMA rate.
        eps (float): A small positive value used for numerical stability.
    """
```

### `__call__(feedback)`

```python
def __call__(self, feedback: float) -> float | None:
    """
    Get the advantage produced by the given feedback.

    Args:
        feedback (float): The feedback.

    Returns:
        float | None: The advantage produced by the given feedback.
    """
```

### Properties

```python
@property
def mean(self) -> float | None:
    """
    Get the mean of the evaluator.

    Returns:
        float | None: The mean of the evaluator.
    """

@property
def var(self) -> float | None:
    """
    Get the variance of the evaluator.

    Returns:
        float | None: The variance of the evaluator.
    """

@property
def mean_ema_rate(self) -> float:
    """
    Get the mean EMA rate of the evaluator.

    Returns:
        float: The mean EMA rate of the evaluator.
    """

@property
def var_ema_rate(self) -> float:
    """
    Get the variance EMA rate of the evaluator.

    Returns:
        float: The variance EMA rate of the evaluator.
    """

@property
def eps(self) -> float:
    """
    Get the epsilon of the evaluator.

    Returns:
        float: The epsilon of the evaluator.
    """
```

### `__eq__(other)`

```python
def __eq__(self, other: object) -> bool:
    """
    Check if two evaluators are equal.

    Args:
        other (object): The object to compare with.

    Returns:
        bool: True if the evaluators are equal, False otherwise.
    """
```

### `to_dict()`

```python
def to_dict(self) -> dict[str, Any]:
    """
    Get the dictionary representation of the evaluator.

    Returns:
        dict[str, Any]: The dictionary representation of the evaluator.
    """
```

### `from_dict(data)` *(classmethod)*

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> ZScoreEvaluator:
    """
    Create an evaluator from a dictionary.

    Args:
        data (dict[str, Any]): The dictionary representation of the evaluator.

    Returns:
        ZScoreEvaluator: The evaluator.
    """
```

### Inherited API

`ZScoreEvaluator` inherits from [`Evaluator`](Evaluator.md).

## Examples

```python
from jacinta.evaluator import ZScoreEvaluator

# Initialize a ZScoreEvaluator
evaluator = ZScoreEvaluator(
    mean_ema_rate=0.1,
    var_ema_rate=0.1,
)

# Warm-up phase: no advantage yet
print(evaluator(0.5))  # None
print(evaluator(0.8))  # None

# From the third call onward
print(evaluator(0.3))  # -0.7666666641111111
print(evaluator(0.9))  # 1.3378650681765694

# Online statistics
print(evaluator.mean)  # 0.5463
print(evaluator.var)   # 0.09310590000000003

# Serialize and deserialize
data = evaluator.to_dict()
evaluator2 = ZScoreEvaluator.from_dict(data)
assert evaluator == evaluator2

# Save and load
evaluator.save("evaluator.json")
evaluator3 = ZScoreEvaluator.load("evaluator.json")
assert evaluator == evaluator3
```

## Limitations

- The first two calls always return `None` while the running statistics are initialized.
- `eps` prevents division by zero when the estimated variance is zero.

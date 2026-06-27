# Transmitter

## Overview

[`Transmitter`](../../../src/jacinta/transmitter/Transmitter.py) is a specialized [`NDSpace`](../utils/ndspace/NDSpace.md) that represents an adaptive probability distribution over a continuous action space. It learns from reward feedback, progressively increasing the probability of sampling actions associated with better outcomes.

Sampling can be biased to favor either more probable or less probable regions, allowing the exploration-exploitation trade-off to be adjusted without altering the learned distribution.

## API Reference

```python
class Transmitter(NDSpace):
    """
    A Transmitter represents an NDSpace that manages the information transmitted
    by a Receiver.

    Attributes:
        log_weight (float): The log-weight of the transmitter.
        evaluator (Evaluator): The evaluator associated to the transmitter.
        bias_scale_scheduler (Scheduler): The bias scale scheduler.
        learning_rate_scheduler (Scheduler): The learning rate scheduler.
        hits_rate_scheduler (Scheduler): The hits rate scheduler.
        hits_left (float): The number of hits left to split the transmitter.
    """
```

### Constructor

```python
def __init__(
    self,
    bounds: tuple[tuple[float, float], ...],
    evaluator: Evaluator,
    bias_scale_scheduler: Scheduler,
    learning_rate_scheduler: Scheduler,
    hits_rate_scheduler: Scheduler,
    min_width: float | None = None,
    max_depth: int | None = None,
    seed: int | None = None,
) -> None:
    """
    Initialize a Transmitter.

    Args:
        bounds (tuple[tuple[float, float], ...]): The bounds of the transmitter.
        evaluator (Evaluator): The evaluator associated to the transmitter.
        bias_scale_scheduler (Scheduler): The bias scale scheduler.
        learning_rate_scheduler (Scheduler): The learning rate scheduler.
        hits_rate_scheduler (Scheduler): The hits rate scheduler.
        min_width (float | None): The minimum width of each dimension of
            the transmitter. Defaults to None.
        max_depth (int | None): The maximum depth of the transmitter.
            Defaults to None.
        seed (int | None): The seed for the random number generator.
            Defaults to None.
    """
```

### Properties

```python
@property
def split_point(self) -> TransmitterSample | None:
    """
    Get the split point of the transmitter.

    Returns:
        TransmitterSample | None: The split point of the transmitter.
    """

@property
def log_weight(self) -> float:
    """
    Get the log-weight of the transmitter.

    Returns:
        float: The log-weight of the transmitter.
    """

@property
def evaluator(self) -> Evaluator:
    """
    Get the evaluator of the transmitter.

    Returns:
        Evaluator: The evaluator of the transmitter.
    """

@property
def bias_scale_scheduler(self) -> Scheduler:
    """
    Get the bias scale scheduler of the transmitter.

    Returns:
        Scheduler: The bias scale scheduler of the transmitter.
    """

@property
def learning_rate_scheduler(self) -> Scheduler:
    """
    Get the learning rate scheduler of the transmitter.

    Returns:
        Scheduler: The learning rate scheduler of the transmitter.
    """

@property
def hits_rate_scheduler(self) -> Scheduler:
    """
    Get the hits rate scheduler of the transmitter.

    Returns:
        Scheduler: The hits rate scheduler of the transmitter.
    """

@property
def hits_left(self) -> float:
    """
    Get the number of hits left in the transmitter.

    Returns:
        float: The number of hits left in the transmitter.
    """
```

### `__eq__(other)`

```python
def __eq__(self, other: object) -> bool:
    """
    Check if two transmitters are equal.

    Args:
        other (object): The object to compare with.

    Returns:
        bool: True if the transmitters are equal, False otherwise.
    """
```

### `forward(bias)`

```python
def forward(self, bias: float = 0.0) -> TransmitterSample:
    """
    Sample a value from the transmitter distribution.

    Args:
        bias (float): The bias to apply to the sampling.
            Defaults to 0.0.

    Returns:
        TransmitterSample: The sampled value.
    """
```

### `backward(tsample, feedback)`

```python
def backward(
    self,
    tsample: TransmitterSample,
    feedback: float,
) -> None:
    """
    Update the transmitter distribution based on the feedback.

    Args:
        tsample (TransmitterSample): The transmitter sample.
        feedback (float): The feedback to apply to the distribution.
    """
```

### `can_split()`

```python
def can_split(self) -> bool:
    """
    Check if the transmitter can be split.

    Returns:
        bool: True if the transmitter can be split, False otherwise.
    """
```

### `split()`

```python
def split(self) -> tuple[Transmitter, ...]:
    """
    Split the transmitter into smaller transmitters.

    Returns:
        tuple[Transmitter, ...]: The sub-transmitters created by the split.
    """
```

### `to_dict()`

```python
def to_dict(self) -> dict[str, Any]:
    """
    Get the dictionary representation of the transmitter.

    Returns:
        dict[str, Any]: The dictionary representation of the transmitter.
    """
```

### `from_dict(data)` *(classmethod)*

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> Transmitter:
    """
    Create a transmitter from a dictionary.

    Args:
        data (dict[str, Any]): The dictionary representation of the transmitter.

    Returns:
        Transmitter: The transmitter.
    """
```

### Inherited API

`Transmitter` inherits from [`NDSpace`](../utils/ndspace/NDSpace.md).

## Examples

```python
import math

from jacinta.evaluator import ZScoreEvaluator
from jacinta.transmitter import Transmitter, TransmitterSample
from jacinta.utils.scheduler import ConstantScheduler

# Initialize a 2D Transmitter
transmitter = Transmitter(
    bounds=((0.0, 10.0), (0.0, 10.0)),
    evaluator=ZScoreEvaluator(mean_ema_rate=0.001, var_ema_rate=0.001),
    bias_scale_scheduler=ConstantScheduler(10.0),
    learning_rate_scheduler=ConstantScheduler(0.001),
    hits_rate_scheduler=ConstantScheduler(10000.0),
)
tsample = transmitter.forward(bias=1.0)
print(tsample.coordinates)  # (6.394267984578837, 0.25010755222666936)

# Reward function (maximum at (5,5))
def get_reward(tsample: TransmitterSample) -> float:
    """
    Reward regions close to the center of the space (5,5).

    Args:
        tsample (TransmitterSample): The transmitter sample.

    Returns:
        float: The reward.
    """
    x, y = tsample.coordinates
    # Calculate the Euclidean distance from the center of the space
    d = math.sqrt((x - 5)**2 + (y - 5)**2)
    # Calculate the maximum possible distance from the center of the space
    d_max = 5 * math.sqrt(2)
    # Normalize the distance to the range [-1, 1]
    reward = 1 - 2 * d / d_max
    return reward

# Forward-backward loop
for _ in range(1000000):
    tsample = transmitter.forward(bias=0.0)
    reward = get_reward(tsample)
    transmitter.backward(tsample, reward)

# Exploit the learned distribution
tsample = transmitter.forward(bias=1.0)
print(tsample.coordinates)  # (5.0000036861057096, 4.999993194516738)

# Serialize and deserialize
data = transmitter.to_dict()
transmitter2 = Transmitter.from_dict(data)
assert transmitter == transmitter2

# Save and load
transmitter.save("transmitter.json")
transmitter3 = Transmitter.load("transmitter.json")
assert transmitter == transmitter3
```

## Limitations

- `backward` does not update the `Transmitter` state if the `Evaluator` returns `None` during its statistics warm-up. For example, `ZScoreEvaluator` requires two observations before producing an advantage.
- `bias` must be in `[-1, 1]`: `-1` favors less probable regions, `0` applies no bias, and `1` favors more probable regions.
- `feedback` must be in `[-1, 1]`: `-1` represents the worst possible outcome, and `1` represents the best possible outcome.
- `collapse` is not implemented. Once a region has been split, it cannot be merged back.

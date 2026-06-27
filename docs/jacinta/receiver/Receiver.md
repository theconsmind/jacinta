# Receiver

## Overview

[`Receiver`](../../../src/jacinta/receiver/Receiver.py) is a specialized [`NDSpace`](../utils/ndspace/NDSpace.md) that represents an adaptive context space. It maps input stimuli to output stimuli by selecting a context-specific [`Transmitter`](../transmitter/Transmitter.md).

As feedback is received, different regions of the context space can develop specialized action distributions, allowing Jacinta to adapt its behavior to different observation patterns.

## API Reference

```python
class Receiver(NDSpace):
    """
    A Receiver represents an NDSpace that manages the information received
    by a Jacinta module.

    Attributes:
        transmitter (Transmitter): The transmitter associated to the receiver.
        hits_rate_scheduler (Scheduler): The hits rate scheduler.
        hits_left (float): The number of hits left to split the receiver.
    """
```

### Constructor

```python
def __init__(
    self,
    bounds: tuple[tuple[float, float], ...],
    transmitter: Transmitter,
    hits_rate_scheduler: Scheduler,
    min_width: float | None = None,
    max_depth: int | None = None,
) -> None:
    """
    Initialize a Receiver.

    Args:
        bounds (tuple[tuple[float, float], ...]): The bounds of the receiver.
        transmitter (Transmitter): The transmitter associated to the receiver.
        hits_rate_scheduler (Scheduler): The hits rate scheduler.
        min_width (float | None): The minimum width of each dimension of
            the receiver. Defaults to None.
        max_depth (int | None): The maximum depth of the receiver.
            Defaults to None.
    """
```

### Properties

```python
@property
def split_point(self) -> ReceiverSample | None:
    """
    Get the split point of the receiver.

    Returns:
        ReceiverSample | None: The split point of the receiver.
    """

@property
def transmitter(self) -> Transmitter:
    """
    Get the transmitter of the receiver.

    Returns:
        Transmitter: The transmitter of the receiver.
    """

@property
def hits_rate_scheduler(self) -> Scheduler:
    """
    Get the hits rate scheduler of the receiver.

    Returns:
        Scheduler: The hits rate scheduler of the receiver.
    """

@property
def hits_left(self) -> float:
    """
    Get the number of hits left in the receiver.

    Returns:
        float: The number of hits left in the receiver.
    """
```

### `__eq__(other)`

```python
def __eq__(self, other: object) -> bool:
    """
    Check if two receivers are equal.

    Args:
        other (object): The object to compare with.

    Returns:
        bool: True if the receivers are equal, False otherwise.
    """
```

### `forward(rsample, bias)`

```python
def forward(self, rsample: ReceiverSample, bias: float = 0.0) -> TransmitterSample:
    """
    Sample a value from the receiver distribution.

    Args:
        rsample (ReceiverSample): The receiver sample.
        bias (float): The bias to apply to the sampling.
            Defaults to 0.0.

    Returns:
        TransmitterSample: The sampled value.
    """
```

### `backward(rsample, tsample, feedback)`

```python
def backward(
    self,
    rsample: ReceiverSample,
    tsample: TransmitterSample,
    feedback: float,
) -> None:
    """
    Update the receiver distribution based on the feedback.

    Args:
        rsample (ReceiverSample): The receiver sample.
        tsample (TransmitterSample): The transmitter sample.
        feedback (float): The feedback to apply to the distribution.
    """
```

### `can_split()`

```python
def can_split(self) -> bool:
    """
    Check if the receiver can be split.

    Returns:
        bool: True if the receiver can be split, False otherwise.
    """
```

### `split()`

```python
def split(self) -> tuple[Receiver, ...]:
    """
    Split the receiver into smaller receivers.

    Returns:
        tuple[Receiver, ...]: The sub-receivers created by the split.
    """
```

### `to_dict()`

```python
def to_dict(self) -> dict[str, Any]:
    """
    Get the dictionary representation of the receiver.

    Returns:
        dict[str, Any]: The dictionary representation of the receiver.
    """
```

### `from_dict(data)` *(classmethod)*

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> Receiver:
    """
    Create a receiver from a dictionary.

    Args:
        data (dict[str, Any]): The dictionary representation of the receiver.

    Returns:
        Receiver: The receiver.
    """
```

### Inherited API

`Receiver` inherits from [`NDSpace`](../utils/ndspace/NDSpace.md).

## Examples

```python
import math

from jacinta.evaluator import ZScoreEvaluator
from jacinta.receiver import Receiver, ReceiverSample
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

# Initialize a 2D Receiver
receiver = Receiver(
    bounds=((0.0, 10.0), (0.0, 10.0)),
    transmitter=transmitter,
    hits_rate_scheduler=ConstantScheduler(10000.0),
)
rsample = ReceiverSample((2.0, 2.0))
tsample = receiver.forward(rsample, bias=1.0)
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
    tsample = receiver.forward(rsample, bias=0.0)
    reward = get_reward(tsample)
    receiver.backward(rsample, tsample, reward)

# Exploit the learned distribution
tsample = receiver.forward(rsample, bias=1.0)
print(tsample.coordinates)  # (5.000001647399828, 4.999992279204593)

# Serialize and deserialize
data = receiver.to_dict()
receiver2 = Receiver.from_dict(data)
assert receiver == receiver2

# Save and load
receiver.save("receiver.json")
receiver3 = Receiver.load("receiver.json")
assert receiver == receiver3
```

## Limitations

- `backward` updates the `Transmitter` of every ancestor from the active `Receiver` node up to the root.
- `collapse` is not implemented. Once a region has been split, it cannot be merged back.

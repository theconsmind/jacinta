# Processor

## Overview

[`Processor`](../../../src/jacinta/processor/Processor.py) is a processing unit that maps input stimuli to output stimuli. It provides an integrated interface for the [`Receiver`](receiver/Receiver.md) and [`Transmitter`](transmitter/Transmitter.md) modules, allowing input-output mappings to be processed without managing either module directly.

Input stimuli can be received independently as they become available, and once all have been received, the `Processor` generates the corresponding input-output stimulus pair.

## API Reference

```python
class Processor:
    """
    A Processor represents a processing unit that maps input coordinates to
    output coordinates.

    Attributes:
        receiver (Receiver): The receiver associated to the processor.
        coordinates (tuple[float | None, ...]): The coordinates of the processor.
    """
```

### Constructor

```python
def __init__(self, receiver: Receiver) -> None:
    """
    Initialize a Processor.

    Args:
        receiver (Receiver): The receiver associated to the processor.
    """
```

### Properties

```python
@property
def receiver(self) -> Receiver:
    """
    Get the receiver of the processor.

    Returns:
        Receiver: The receiver of the processor.
    """

@property
def coordinates(self) -> tuple[float | None, ...]:
    """
    Get the coordinates of the processor.

    Returns:
        tuple[float | None, ...]: The coordinates of the processor.
    """

@property
def rbounds(self) -> tuple[tuple[float, float], ...]:
    """
    Get the bounds of the receiver.

    Returns:
        tuple[tuple[float, float], ...]: The bounds of the receiver.
    """

@property
def tbounds(self) -> tuple[tuple[float, float], ...]:
    """
    Get the bounds of the transmitter.

    Returns:
        tuple[tuple[float, float], ...]: The bounds of the transmitter.
    """

@property
def rnd(self) -> int:
    """
    Get the number of dimensions of the receiver.

    Returns:
        int: The number of dimensions of the receiver.
    """

@property
def tnd(self) -> int:
    """
    Get the number of dimensions of the transmitter.

    Returns:
        int: The number of dimensions of the transmitter.
    """

@property
def is_ready(self) -> bool:
    """
    Check if all coordinates have been received.

    Returns:
        bool: True if all coordinates have been received, False otherwise.
    """
```

### `__eq__(other)`

```python
def __eq__(self, other: object) -> bool:
    """
    Check if two processors are equal.

    Args:
        other (object): The object to compare with.

    Returns:
        bool: True if the processors are equal, False otherwise.
    """
```

### `receive(idx, coord)`

```python
def receive(self, idx: int, coord: float) -> None:
    """
    Receive a coordinate.

    Args:
        idx (int): The index of the coordinate.
        coord (float): The coordinate to receive.
    """
```

### `forward(bias=0.0)`

```python
def forward(self, bias: float = 0.0) -> ProcessorSample:
    """
    Sample a value for the received coordinates.

    Args:
        bias (float): The bias to apply to the sampling.
            Defaults to 0.0.

    Returns:
        ProcessorSample: The sampled value.
    """
```

### `backward(psample, feedback)`

```python
def backward(self, psample: ProcessorSample, feedback: float) -> None:
    """
    Update the input-output mapping distribution based on the feedback.

    Args:
        psample (ProcessorSample): The processor sample.
        feedback (float): The feedback to apply to the distribution.
    """
```

### `add_rdimensions(bounds)`

```python
def add_rdimensions(self, bounds: tuple[tuple[float, float], ...]) -> None:
    """
    Add new dimensions to the receiver.

    Args:
        bounds (tuple[tuple[float, float], ...]): The bounds of the new dimensions.
    """
```

### `add_tdimensions(bounds)`

```python
def add_tdimensions(self, bounds: tuple[tuple[float, float], ...]) -> None:
    """
    Add new dimensions to the transmitter.

    Args:
        bounds (tuple[tuple[float, float], ...]): The bounds of the new dimensions.
    """
```

### `remove_rdimensions(dims)`

```python
def remove_rdimensions(self, dims: set[int]) -> None:
    """
    Remove dimensions from the receiver.

    Args:
        dims (set[int]): The indices of the dimensions to remove.
    """
```

### `remove_tdimensions(dims)`

```python
def remove_tdimensions(self, dims: set[int]) -> None:
    """
    Remove dimensions from the transmitter.

    Args:
        dims (set[int]): The indices of the dimensions to remove.
    """
```

### `copy()`

```python
def copy(self) -> Processor:
    """
    Get a copy of the processor.

    Returns:
        Processor: The copy of the processor.
    """
```

### `to_dict()`

```python
def to_dict(self) -> dict[str, Any]:
    """
    Get the dictionary representation of the processor.

    Returns:
        dict[str, Any]: The dictionary representation of the processor.
    """
```

### `from_dict(data)` *(classmethod)*

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> Processor:
    """
    Create a processor from a dictionary.

    Args:
        data (dict[str, Any]): The dictionary representation of
            the processor.

    Returns:
        Processor: The processor.
    """
```

### `save(path, overwrite=False)`

```python
def save(self, path: str | Path, overwrite: bool = False) -> None:
    """
    Save the processor to a json file.

    Args:
        path (str | Path): The path to the file.
        overwrite (bool): Whether to overwrite the file if it exists.
            Defaults to False.
    """
```

### `load(path)` *(classmethod)*

```python
@classmethod
def load(cls, path: str | Path) -> Processor:
    """
    Load a processor from a json file.

    Args:
        path (str | Path): The path to the file.

    Returns:
        Processor: The processor.
    """
```

## Examples

```python
import math

from jacinta.processor import Processor, ProcessorSample
from jacinta.processor.evaluator import ZScoreEvaluator
from jacinta.processor.receiver import Receiver
from jacinta.processor.transmitter import Transmitter
from jacinta.utils.scheduler import ConstantScheduler

# Initialize a 3D Transmitter
transmitter = Transmitter(
    bounds=((0.0, 10.0), (0.0, 10.0), (0.0, 10.0)),
    evaluator=ZScoreEvaluator(mean_ema_rate=0.001, var_ema_rate=0.001),
    bias_scale_scheduler=ConstantScheduler(value=10.0),
    learning_rate_scheduler=ConstantScheduler(value=0.001),
    hits_rate_scheduler=ConstantScheduler(value=10000.0),
)

# Initialize a 2D Receiver
receiver = Receiver(
    bounds=((0.0, 10.0), (0.0, 10.0)),
    transmitter=transmitter,
    hits_rate_scheduler=ConstantScheduler(value=10000.0),
)

# Initialize a 2D-to-3D Processor
processor = Processor(receiver)

# Receive input coordinates
processor.receive(idx=0, coord=2.0)
processor.receive(idx=1, coord=2.0)
assert processor.is_ready

psample = processor.forward(bias=1.0)
print(psample.rcoordinates)  # (2.0, 2.0)
print(psample.tcoordinates)  # (1.208366262746875, 3.5199732867667786, 8.9084225479461)

# Reward function (maximum at (5,5,5))
def get_reward(psample: ProcessorSample) -> float:
    """
    Reward regions close to the center of the space (5,5,5).

    Args:
        psample (ProcessorSample): The processor sample.

    Returns:
        float: The reward.
    """
    x, y, z = psample.tcoordinates
    # Calculate the Euclidean distance from the center of the space
    d = math.sqrt((x - 5)**2 + (y - 5)**2 + (z - 5)**2)
    # Calculate the maximum possible distance from the center of the space
    d_max = 5 * math.sqrt(3)
    # Normalize the distance to the range [-1, 1]
    reward = 1 - 2 * d / d_max
    return reward

# Forward-backward loop
for _ in range(1000000):
    processor.receive(idx=0, coord=2.0)
    processor.receive(idx=1, coord=2.0)
    psample = processor.forward(bias=0.0)
    reward = get_reward(psample)
    processor.backward(psample, reward)

# Exploit the learned distribution
processor.receive(idx=0, coord=2.0)
processor.receive(idx=1, coord=2.0)
psample = processor.forward(bias=1.0)
print(psample.rcoordinates)  # (2.0, 2.0)
print(psample.tcoordinates)  # (5.071257298266986, 5.061971945896996, 5.00112896188753)

# Serialize and deserialize
data = processor.to_dict()
processor2 = Processor.from_dict(data)
assert processor == processor2

# Save and load
processor.save(path="processor.json")
processor3 = Processor.load(path="processor.json")
assert processor == processor3
```

## Limitations

- `forward` cannot be called until all input coordinates have been received.
- `receive` cannot be used for a new input stimulus until the current one has been consumed by `forward`.

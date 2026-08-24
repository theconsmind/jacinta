# ProcessorSample

## Overview

[`ProcessorSample`](../../../src/jacinta/processor/processor_sample.py) represents an input-output stimulus pair used by the [`Processor`](processor.md) module.

## API Reference

```python
class ProcessorSample:
    """
    A ProcessorSample represents an input-output coordinate mapping produced
    by a Processor.

    Attributes:
        rsample (ReceiverSample): The receiver sample of the processor sample.
        tsample (TransmitterSample): The transmitter sample of the processor sample.
    """
```

### Constructor

```python
def __init__(self, rsample: ReceiverSample, tsample: TransmitterSample) -> None:
    """
    Initialize a ProcessorSample.

    Args:
        rsample (ReceiverSample): The receiver sample of the processor sample.
        tsample (TransmitterSample): The transmitter sample of the processor sample.
    """
```

### Properties

```python
@property
def rsample(self) -> ReceiverSample:
    """
    Get the receiver sample of the processor sample.

    Returns:
        ReceiverSample: The receiver sample of the processor sample.
    """

@property
def tsample(self) -> TransmitterSample:
    """
    Get the transmitter sample of the processor sample.

    Returns:
        TransmitterSample: The transmitter sample of the processor sample.
    """

@property
def rcoordinates(self) -> tuple[float, ...]:
    """
    Get the coordinates of the receiver sample.

    Returns:
        tuple[float, ...]: The coordinates of the receiver sample.
    """

@property
def tcoordinates(self) -> tuple[float, ...]:
    """
    Get the coordinates of the transmitter sample.

    Returns:
        tuple[float, ...]: The coordinates of the transmitter sample.
    """

@property
def rnd(self) -> int:
    """
    Get the number of dimensions of the receiver sample.

    Returns:
        int: The number of dimensions of the receiver sample.
    """

@property
def tnd(self) -> int:
    """
    Get the number of dimensions of the transmitter sample.

    Returns:
        int: The number of dimensions of the transmitter sample.
    """
```

### `__eq__(other)`

```python
def __eq__(self, other: object) -> bool:
    """
    Check if two processor samples are equal.

    Args:
        other (object): The object to compare with.

    Returns:
        bool: True if the processor samples are equal, False otherwise.
    """
```

### `copy()`

```python
def copy(self) -> ProcessorSample:
    """
    Get a copy of the processor sample.

    Returns:
        ProcessorSample: The copy of the processor sample.
    """
```

### `to_dict()`

```python
def to_dict(self) -> dict[str, Any]:
    """
    Get the dictionary representation of the processor sample.

    Returns:
        dict[str, Any]: The dictionary representation of the processor sample.
    """
```

### `from_dict(data)` *(classmethod)*

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> ProcessorSample:
    """
    Create a processor sample from a dictionary.

    Args:
        data (dict[str, Any]): The dictionary representation of
            the processor sample.

    Returns:
        ProcessorSample: The processor sample.
    """
```

### `save(path, overwrite=False)`

```python
def save(self, path: str | Path, overwrite: bool = False) -> None:
    """
    Save the processor sample to a json file.

    Args:
        path (str | Path): The path to the file.
        overwrite (bool): Whether to overwrite the file if it exists.
            Defaults to False.
    """
```

### `load(path)` *(classmethod)*

```python
@classmethod
def load(cls, path: str | Path) -> ProcessorSample:
    """
    Load a processor sample from a json file.

    Args:
        path (str | Path): The path to the file.

    Returns:
        ProcessorSample: The processor sample.
    """
```

## Examples

```python
from jacinta.processor import ProcessorSample
from jacinta.processor.receiver import ReceiverSample
from jacinta.processor.transmitter import TransmitterSample

# initialize a 2D-to-3D ProcessorSample
psample = ProcessorSample(
    rsample=ReceiverSample(coordinates=(0.1, 0.5)),
    tsample=TransmitterSample(coordinates=(0.3, 0.7, 0.9)),
)
print(psample.rnd)           # 2
print(psample.tnd)           # 3
print(psample.rcoordinates)  # (0.1, 0.5)
print(psample.tcoordinates)  # (0.3, 0.7, 0.9)

# serialize and deserialize
data = psample.to_dict()
psample2 = ProcessorSample.from_dict(data)
assert psample == psample2

# save and load
psample.save(path="psample.json")
psample3 = ProcessorSample.load(path="psample.json")
assert psample == psample3
```

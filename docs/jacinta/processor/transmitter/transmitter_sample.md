# TransmitterSample

## Overview

[`TransmitterSample`](../../../../src/jacinta/processor/transmitter/transmitter_sample.py) is a specialized [`NDPoint`](../../utils/ndspace/nd_point.md) used by the [`Transmitter`](transmitter.md) module to represent output stimuli.

Although it shares the same N-dimensional coordinate representation as `NDPoint`, it provides a distinct semantic type that allows the `Transmitter` module to differentiate output stimuli from other spatial representations used throughout Jacinta.

## API Reference

```python
class TransmitterSample(NDPoint):
    """
    A TransmitterSample represents an NDPoint transmitted by a Transmitter.
    """
```

### Constructor

```python
def __init__(self, coordinates: tuple[float, ...]) -> None:
    """
    Initialize a TransmitterSample.

    Args:
        coordinates (tuple[float, ...]): The coordinates of the point.
    """
```

### Inherited API

`TransmitterSample` inherits from [`NDPoint`](../../utils/ndspace/nd_point.md).

## Examples

```python
from jacinta.processor.transmitter import TransmitterSample

# initialize a 3D TransmitterSample
tsample = TransmitterSample(
    coordinates=(0.3, 0.7, 0.9),
)
print(tsample.nd)           # 3
print(tsample.coordinates)  # (0.3, 0.7, 0.9)

# serialize and deserialize
data = tsample.to_dict()
tsample2 = TransmitterSample.from_dict(data)
assert tsample == tsample2

# save and load
tsample.save(path="tsample.json")
tsample3 = TransmitterSample.load(path="tsample.json")
assert tsample == tsample3
```

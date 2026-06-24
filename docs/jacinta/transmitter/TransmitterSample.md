# TransmitterSample

## Overview

[`TransmitterSample`](../../../src/jacinta/transmitter/TransmitterSample.py) is a specialized [`NDPoint`](../utils/ndspace/NDPoint.md) used by the [`Transmitter`](Transmitter.md) module to represent output stimuli.

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

`TransmitterSample` inherits from [`NDPoint`](../utils/ndspace/NDPoint.md).

## Examples

```python
from jacinta.transmitter import TransmitterSample

# Initialize a 3D TransmitterSample
tsample = TransmitterSample((0.1, 0.5, 0.9))
print(tsample.nd)           # 3
print(tsample.coordinates)  # (0.1, 0.5, 0.9)

# Serialize and deserialize
data = tsample.to_dict()
tsample2 = TransmitterSample.from_dict(data)
assert tsample == tsample2

# Save and load
tsample.save("tsample.json")
tsample3 = TransmitterSample.load("tsample.json")
assert tsample == tsample3
```

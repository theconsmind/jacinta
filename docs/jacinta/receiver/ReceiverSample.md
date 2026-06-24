# ReceiverSample

## Overview

[`ReceiverSample`](../../../src/jacinta/receiver/ReceiverSample.py) is a specialized [`NDPoint`](../utils/ndspace/NDPoint.md) used by the [`Receiver`](Receiver.md) module to represent input stimuli.

Although it shares the same N-dimensional coordinate representation as `NDPoint`, it provides a distinct semantic type that allows the `Receiver` module to differentiate input stimuli from other spatial representations used throughout Jacinta.

## API Reference

```python
class ReceiverSample(NDPoint):
    """
    A ReceiverSample represents an NDPoint received by a Receiver.
    """
```

### Constructor

```python
def __init__(self, coordinates: tuple[float, ...]) -> None:
    """
    Initialize a ReceiverSample.

    Args:
        coordinates (tuple[float, ...]): The coordinates of the point.
    """
```

### Inherited API

`ReceiverSample` inherits from [`NDPoint`](../utils/ndspace/NDPoint.md).

## Examples

```python
from jacinta.receiver import ReceiverSample

# Initialize a 3D ReceiverSample
rsample = ReceiverSample((0.1, 0.5, 0.9))
print(rsample.nd)           # 3
print(rsample.coordinates)  # (0.1, 0.5, 0.9)

# Serialize and deserialize
data = rsample.to_dict()
rsample2 = ReceiverSample.from_dict(data)
assert rsample == rsample2

# Save and load
rsample.save("rsample.json")
rsample3 = ReceiverSample.load("rsample.json")
assert rsample == rsample3
```

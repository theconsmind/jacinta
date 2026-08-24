# ReceiverSample

## Overview

[`ReceiverSample`](../../../../src/jacinta/processor/receiver/receiver_sample.py) is a specialized [`NDPoint`](../../utils/ndspace/nd_point.md) used by the [`Receiver`](receiver.md) module to represent input stimuli.

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

`ReceiverSample` inherits from [`NDPoint`](../../utils/ndspace/nd_point.md).

## Examples

```python
from jacinta.processor.receiver import ReceiverSample

# initialize a 2D ReceiverSample
rsample = ReceiverSample(
    coordinates=(0.1, 0.5),
)
print(rsample.nd)           # 2
print(rsample.coordinates)  # (0.1, 0.5)

# serialize and deserialize
data = rsample.to_dict()
rsample2 = ReceiverSample.from_dict(data)
assert rsample == rsample2

# save and load
rsample.save(path="rsample.json")
rsample3 = ReceiverSample.load(path="rsample.json")
assert rsample == rsample3
```

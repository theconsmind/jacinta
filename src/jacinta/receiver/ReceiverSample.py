from __future__ import annotations

from ..utils.ndspace import NDPoint


class ReceiverSample(NDPoint):
    """
    A ReceiverSample represents an NDPoint received by a Receiver.
    """

    def __init__(self, coordinates: tuple[float, ...]) -> None:
        """
        Initialize a ReceiverSample.

        Args:
            coordinates (tuple[float, ...]): The coordinates of the point.
        """
        super().__init__(coordinates)
        return

from __future__ import annotations

from ..utils.ndspace import NDPoint


class TransmitterSample(NDPoint):
    """
    A TransmitterSample represents a sample returned by the Transmitter.
    """

    def __init__(self, coordinates: tuple[float, ...]) -> None:
        """
        Initialize a TransmitterSample.

        Args:
            coordinates (tuple[float, ...]): The coordinates of the TransmitterSample.
        """
        super().__init__(coordinates)
        return

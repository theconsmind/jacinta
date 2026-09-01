from __future__ import annotations

from typing import Any

from ...utils.ndspace import NDPoint


class TransmitterSample(NDPoint):
    """
    A TransmitterSample represents an NDPoint transmitted by a Transmitter.
    """

    def __init__(self, coordinates: tuple[float, ...]) -> None:
        """
        Initialize a TransmitterSample.

        Args:
            coordinates (tuple[float, ...]): The coordinates of the point.
        """
        super().__init__(coordinates)
        return

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransmitterSample:
        """
        Create a transmitter sample from a dictionary.

        Args:
            data (dict[str, Any]): The dictionary representation
                of the transmitter sample.

        Returns:
            TransmitterSample: The transmitter sample.
        """
        # data validations
        if not isinstance(data, dict):
            raise TypeError("data must be a dict.")
        if "type" not in data:
            raise KeyError("data must contain the key 'type'.")
        if data["type"] != cls.__name__:
            raise ValueError(f"data['type'] must be a {cls.__name__}.")
        if "coordinates" not in data:
            raise KeyError("data must contain the key 'coordinates'.")
        # initializations
        tsample = cls(data["coordinates"])
        return tsample

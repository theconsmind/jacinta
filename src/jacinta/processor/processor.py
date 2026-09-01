from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .processor_sample import ProcessorSample
from .receiver import Receiver
from .transmitter import TransmitterSample


class Processor:
    """
    A Processor represents a processing unit that maps input coordinates to
    output coordinates.

    Attributes:
        receiver (Receiver): The receiver associated to the processor.
    """

    __slots__ = (
        "_receiver",
        "_frozen",
    )

    def __init__(self, receiver: Receiver) -> None:
        """
        Initialize a Processor.

        Args:
            receiver (Receiver): The receiver associated to the processor.
        """
        # receiver validations
        if not isinstance(receiver, Receiver):
            raise TypeError("receiver must be a Receiver.")
        # initializations
        object.__setattr__(self, "_frozen", False)
        self._receiver = receiver
        object.__setattr__(self, "_frozen", True)
        return

    def __repr__(self) -> str:
        """
        Get the representation of the processor.

        Returns:
            str: The representation of the processor.
        """
        processor = f"{self.__class__.__name__}(receiver={self._receiver!r})"
        return processor

    @property
    def receiver(self) -> Receiver:
        """
        Get the receiver of the processor.

        Returns:
            Receiver: The receiver of the processor.
        """
        return self._receiver

    @property
    def rbounds(self) -> tuple[tuple[float, float], ...]:
        """
        Get the bounds of the receiver.

        Returns:
            tuple[tuple[float, float], ...]: The bounds of the receiver.
        """
        return self._receiver.bounds

    @property
    def tbounds(self) -> tuple[tuple[float, float], ...]:
        """
        Get the bounds of the transmitter.

        Returns:
            tuple[tuple[float, float], ...]: The bounds of the transmitter.
        """
        return self._receiver.transmitter.bounds

    @property
    def rnd(self) -> int:
        """
        Get the number of dimensions of the receiver.

        Returns:
            int: The number of dimensions of the receiver.
        """
        return self._receiver.nd

    @property
    def tnd(self) -> int:
        """
        Get the number of dimensions of the transmitter.

        Returns:
            int: The number of dimensions of the transmitter.
        """
        return self._receiver.transmitter.nd

    def __eq__(self, other: object) -> bool:
        """
        Check if two processors are equal.

        Args:
            other (object): The object to compare with.

        Returns:
            bool: True if the processors are equal, False otherwise.
        """
        # other validations
        if type(self) is not type(other):
            return NotImplemented
        # equality check
        is_equal = self._receiver == other._receiver
        return is_equal

    def forward(self, psample: ProcessorSample, bias: float = 0.0) -> ProcessorSample:
        """
        Sample a value from the input-output mapping distribution.

        Args:
            psample (ProcessorSample): The processor sample.
            bias (float): The bias to apply to the sampling.
                Defaults to 0.0.

        Returns:
            ProcessorSample: The sampled value.
        """
        # psample validations
        if not isinstance(psample, ProcessorSample):
            raise TypeError("psample must be a ProcessorSample.")
        if psample.tsample is not None:
            raise ValueError("psample.tsample must be None.")
        # generate the processor tsample
        tsample = self._receiver.forward(psample.rsample, bias)
        psample = ProcessorSample(psample.rsample, tsample)
        return psample

    def backward(self, psample: ProcessorSample, feedback: float) -> None:
        """
        Update the input-output mapping distribution based on the feedback.

        Args:
            psample (ProcessorSample): The processor sample.
            feedback (float): The feedback to apply to the distribution.
        """
        # psample validations
        if not isinstance(psample, ProcessorSample):
            raise TypeError("psample must be a ProcessorSample.")
        if not isinstance(psample.tsample, TransmitterSample):
            raise TypeError("psample.tsample must be a TransmitterSample.")
        # propagate the feedback
        self._receiver.backward(psample.rsample, psample.tsample, feedback)
        return

    def add_rdimensions(self, bounds: tuple[tuple[float, float], ...]) -> None:
        """
        Add new dimensions to the receiver.

        Args:
            bounds (tuple[tuple[float, float], ...]): The bounds of the new dimensions.
        """
        self._receiver.add_dimensions(bounds)
        return

    def remove_rdimensions(self, dims: set[int]) -> None:
        """
        Remove dimensions from the receiver.

        Args:
            dims (set[int]): The indices of the dimensions to remove.
        """
        self._receiver.remove_dimensions(dims)
        return

    def add_tdimensions(self, bounds: tuple[tuple[float, float], ...]) -> None:
        """
        Add new dimensions to the transmitter.

        Args:
            bounds (tuple[tuple[float, float], ...]): The bounds of the new dimensions.
        """
        receivers = [self._receiver.root]
        while receivers:
            receiver = receivers.pop()
            transmitter = receiver.transmitter.root
            transmitter.add_dimensions(bounds)
            if receiver.children is not None:
                receivers.extend(receiver.children)
        return

    def remove_tdimensions(self, dims: set[int]) -> None:
        """
        Remove dimensions from the transmitter.

        Args:
            dims (set[int]): The indices of the dimensions to remove.
        """
        receivers = [self._receiver.root]
        while receivers:
            receiver = receivers.pop()
            transmitter = receiver.transmitter.root
            transmitter.remove_dimensions(dims)
            if receiver.children is not None:
                receivers.extend(receiver.children)
        return

    def copy(self) -> Processor:
        """
        Get a copy of the processor.

        Returns:
            Processor: The copy of the processor.
        """
        processor = deepcopy(self)
        return processor

    def to_dict(self) -> dict[str, Any]:
        """
        Get the dictionary representation of the processor.

        Returns:
            dict[str, Any]: The dictionary representation of the processor.
        """
        processor = {
            "type": self.__class__.__name__,
            "receiver": self._receiver.to_dict(),
        }
        return processor

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
        # data validations
        if not isinstance(data, dict):
            raise TypeError("data must be a dict.")
        if "type" not in data:
            raise KeyError("data must contain the key 'type'.")
        if data["type"] != cls.__name__:
            raise ValueError(f"data['type'] must be a {cls.__name__}.")
        if "receiver" not in data:
            raise KeyError("data must contain the key 'receiver'.")
        # initializations
        processor = cls(
            Receiver.from_dict(data["receiver"]),
        )
        return processor

    def save(self, path: str | Path, overwrite: bool = False) -> None:
        """
        Save the processor to a json file.

        Args:
            path (str | Path): The path to the file.
            overwrite (bool): Whether to overwrite the file if it exists.
                Defaults to False.
        """
        # path validations
        if not isinstance(path, (str, Path)):
            raise TypeError("path must be a string or a Path.")
        # file validations
        path = Path(path)
        if path.suffix != ".json":
            raise ValueError("path must have a .json extension.")
        if not overwrite and path.exists():
            raise FileExistsError(f"path already exists: {path}.")
        # file creation
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=4)
        return

    @classmethod
    def load(cls, path: str | Path) -> Processor:
        """
        Load a processor from a json file.

        Args:
            path (str | Path): The path to the file.

        Returns:
            Processor: The processor.
        """
        # path validations
        if not isinstance(path, (str, Path)):
            raise TypeError("path must be a string or a Path.")
        # file validations
        path = Path(path)
        if path.suffix != ".json":
            raise ValueError("path must have a .json extension.")
        if not path.exists():
            raise FileNotFoundError(f"path does not exist: {path}.")
        # file loading
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        processor = cls.from_dict(data)
        return processor

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Set an attribute of the processor.

        Args:
            name (str): The name of the attribute.
            value (Any): The value of the attribute.
        """
        # freeze check
        if getattr(self, "_frozen", False):
            raise AttributeError(f"{self.__class__.__name__} is immutable.")
        # set the attribute
        object.__setattr__(self, name, value)
        return

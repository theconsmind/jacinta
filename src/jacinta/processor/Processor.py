from __future__ import annotations

from typing import Any

from .ProcessorSample import ProcessorSample
from .receiver import Receiver, ReceiverSample


class Processor:
    """ """

    __slots__ = (
        "_receiver",
        "_coordinates",
        "_frozen",
    )

    def __init__(self, receiver: Receiver) -> None:
        """ """
        # receiver validations
        if not isinstance(receiver, Receiver):
            raise TypeError("receiver must be a Receiver.")
        # initializations
        object.__setattr__(self, "_frozen", False)
        self._receiver = receiver
        self._coordinates = (None,) * receiver.nd
        object.__setattr__(self, "_frozen", True)
        return

    def __repr__(self) -> str:
        """ """
        result = (
            f"{self.__class__.__name__}"
            f"(receiver={self._receiver!r}, "
            f"coordinates={self._coordinates!r})"
        )
        return result

    @property
    def receiver(self) -> Receiver:
        """ """
        return self._receiver

    @property
    def coordinates(self) -> tuple[float | None, ...]:
        """ """
        return self._coordinates

    @property
    def rnd(self) -> int:
        """ """
        return self._receiver.nd

    @property
    def tnd(self) -> int:
        """ """
        return self._receiver.transmitter.nd

    @property
    def is_ready(self) -> bool:
        """ """
        is_ready = all(coord is not None for coord in self.coordinates)
        return is_ready

    def receive(self, idx: int, coord: float) -> None:
        """ """
        # idx validations
        if not isinstance(idx, int):
            raise TypeError("idx must be an int.")
        if not (0 <= idx < self.rnd):
            raise IndexError(f"idx must be in [0, {self.rnd}).")
        if self._coordinates[idx] is not None:
            raise RuntimeError(f"coord {idx} has already been received.")
        # coord validations
        if not isinstance(coord, (float, int)):
            raise TypeError("coord must be a float.")
        # register the coordinate
        coords = list(self._coordinates)
        coords[idx] = float(coord)
        object.__setattr__(self, "_frozen", False)
        self._coordinates = tuple(coords)
        object.__setattr__(self, "_frozen", True)
        return

    def forward(self, bias: float = 0.0) -> ProcessorSample:
        """ """
        # bias validations
        if not isinstance(bias, (float, int)):
            raise TypeError("bias must be a float.")
        # processor validations
        if not self.is_ready:
            raise RuntimeError("self is not ready.")
        # generate the processor sample
        rsample = ReceiverSample(self._coordinates)
        tsample = self._receiver.forward(rsample, float(bias))
        psample = ProcessorSample(rsample, tsample)
        # reset the coordinates
        object.__setattr__(self, "_frozen", False)
        self._coordinates = (None,) * self.rnd
        object.__setattr__(self, "_frozen", True)
        return psample

    def backward(self, psample: ProcessorSample, feedback: float) -> None:
        """ """
        # psample validations
        if not isinstance(psample, ProcessorSample):
            raise TypeError("psample must be a ProcessorSample.")
        # feedback validations
        if not isinstance(feedback, (float, int)):
            raise TypeError("feedback must be a float.")
        # propagate the feedback
        self._receiver.backward(psample.rsample, psample.tsample, float(feedback))
        return

    def __setattr__(self, name: str, value: Any) -> None:
        """ """
        # freeze check
        if getattr(self, "_frozen", False):
            raise AttributeError(f"{self.__class__.__name__} is immutable.")
        # set the attribute
        object.__setattr__(self, name, value)
        return

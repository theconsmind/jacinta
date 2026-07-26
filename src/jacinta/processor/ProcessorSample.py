from __future__ import annotations

from typing import Any

from .receiver import ReceiverSample
from .transmitter import TransmitterSample


class ProcessorSample:
    """ """

    __slots__ = (
        "_rsample",
        "_tsample",
        "_frozen",
    )

    def __init__(self, rsample: ReceiverSample, tsample: TransmitterSample) -> None:
        """ """
        # rsample validations
        if not isinstance(rsample, ReceiverSample):
            raise TypeError("rsample must be a ReceiverSample.")
        # tsample validations
        if not isinstance(tsample, TransmitterSample):
            raise TypeError("tsample must be a TransmitterSample.")
        # initializations
        object.__setattr__(self, "_frozen", False)
        self._rsample = rsample
        self._tsample = tsample
        object.__setattr__(self, "_frozen", True)
        return

    @property
    def rsample(self) -> ReceiverSample:
        """ """
        return self._rsample

    @property
    def tsample(self) -> TransmitterSample:
        """ """
        return self._tsample

    def __setattr__(self, name: str, value: Any) -> None:
        """ """
        # freeze check
        if getattr(self, "_frozen", False):
            raise AttributeError(f"{self.__class__.__name__} is immutable.")
        # set the attribute
        object.__setattr__(self, name, value)
        return

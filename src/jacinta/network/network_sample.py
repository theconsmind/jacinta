from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..processor import ProcessorSample
from ..processor.receiver import ReceiverSample
from ..processor.transmitter import TransmitterSample


class NetworkSample:
    """ """

    __slots__ = (
        "_rsample",
        "_tsample",
        "_psamples",
        "_frozen",
    )

    def __init__(
        self,
        rsample: ReceiverSample,
        psamples: tuple[ProcessorSample, ...],
        tsample: TransmitterSample | None = None,
    ) -> None:
        """ """
        # rsample validations
        if not isinstance(rsample, ReceiverSample):
            raise TypeError("rsample must be a ReceiverSample.")
        # tsample validations
        if tsample is not None:
            if not isinstance(tsample, TransmitterSample):
                raise TypeError("tsample must be a TransmitterSample.")
        # psamples validations
        if not isinstance(psamples, (tuple, list)):
            raise TypeError("psamples must be a tuple.")
        if not all(isinstance(psample, ProcessorSample) for psample in psamples):
            raise TypeError("All psamples must be ProcessorSamples.")
        # initializations
        object.__setattr__(self, "_frozen", False)
        self._rsample = rsample
        self._tsample = tsample
        self._psamples = tuple(psamples)
        object.__setattr__(self, "_frozen", True)
        return

    def __repr__(self) -> str:
        """ """
        nsample = (
            f"{self.__class__.__name__}"
            f"(rsample={self._rsample!r}, tsample={self._tsample!r}, "
            f"psamples={self._psamples!r})"
        )
        return nsample

    @property
    def rsample(self) -> ReceiverSample:
        """ """
        return self._rsample

    @property
    def tsample(self) -> TransmitterSample | None:
        """ """
        return self._tsample

    @property
    def psamples(self) -> tuple[ProcessorSample, ...]:
        """ """
        return self._psamples

    @property
    def rcoordinates(self) -> tuple[float, ...]:
        """ """
        return self._rsample.coordinates

    @property
    def tcoordinates(self) -> tuple[float, ...] | None:
        """ """
        tcoordinates = self._tsample.coordinates if self._tsample is not None else None
        return tcoordinates

    @property
    def rnd(self) -> int:
        """ """
        return self._rsample.nd

    @property
    def tnd(self) -> int | None:
        """ """
        tnd = self._tsample.nd if self._tsample is not None else None
        return tnd

    def __eq__(self, other: object) -> bool:
        """ """
        # other validations
        if type(self) is not type(other):
            return NotImplemented
        # equality check
        is_equal = (
            self._rsample == other._rsample
            and self._tsample == other._tsample
            and self._psamples == other._psamples
        )
        return is_equal

    def copy(self) -> NetworkSample:
        """ """
        nsample = deepcopy(self)
        return nsample

    def to_dict(self) -> dict[str, Any]:
        """ """
        nsample = {
            "type": self.__class__.__name__,
            "rsample": self._rsample.to_dict(),
            "psamples": tuple(psample.to_dict() for psample in self._psamples),
        }
        # add tsample
        if self._tsample is not None:
            nsample["tsample"] = self._tsample.to_dict()
        return nsample

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NetworkSample:
        """ """
        # data validations
        if not isinstance(data, dict):
            raise TypeError("data must be a dict.")
        if "type" not in data:
            raise KeyError("data must contain the key 'type'.")
        if data["type"] != cls.__name__:
            raise ValueError(f"data['type'] must be a {cls.__name__}.")
        if "rsample" not in data:
            raise KeyError("data must contain the key 'rsample'.")
        if "psamples" not in data:
            raise KeyError("data must contain the key 'psamples'.")
        if not isinstance(data["psamples"], (tuple, list)):
            raise TypeError("data['psamples'] must be a tuple.")
        # initializations
        psamples = tuple(
            ProcessorSample.from_dict(psample_data) for psample_data in data["psamples"]
        )
        nsample = cls(
            ReceiverSample.from_dict(data["rsample"]),
            psamples,
            TransmitterSample.from_dict(data["tsample"])
            if data.get("tsample") is not None
            else None,
        )
        return nsample

    def save(self, path: str | Path, overwrite: bool = False) -> None:
        """ """
        # path validations
        if not isinstance(path, (str, Path)):
            raise TypeError("path must be a string or a Path.")
        # overwrite validations
        if not isinstance(overwrite, bool):
            raise TypeError("overwrite must be a bool.")
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
    def load(cls, path: str | Path) -> NetworkSample:
        """ """
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
        nsample = cls.from_dict(data)
        return nsample

    def __setattr__(self, name: str, value: Any) -> None:
        """ """
        # freeze check
        if getattr(self, "_frozen", False):
            raise AttributeError(f"{self.__class__.__name__} is immutable.")
        # set the attribute
        object.__setattr__(self, name, value)
        return

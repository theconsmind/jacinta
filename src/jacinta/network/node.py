from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..processor import Processor


class Node:
    """ """

    __slots__ = (
        "_processor",
        "_defaults",
        "_frozen",
    )

    def __init__(self, processor: Processor, defaults: tuple[float, ...]) -> None:
        """ """
        # processor validations
        if not isinstance(processor, Processor):
            raise TypeError("processor must be a Processor.")
        # defaults validations
        if not isinstance(defaults, (tuple, list)):
            raise TypeError("defaults must be a tuple.")
        if len(defaults) != processor.rnd:
            raise ValueError(f"defaults must have length {processor.rnd}.")
        for default, (lower, upper) in zip(defaults, processor.rbounds, strict=True):
            if not isinstance(default, (float, int)):
                raise TypeError("All defaults must be floats.")
            if not (lower <= default < upper):
                raise ValueError("All defaults must be contained in processor bounds.")
        # initializations
        object.__setattr__(self, "_frozen", False)
        self._processor = processor
        self._defaults = tuple(float(default) for default in defaults)
        object.__setattr__(self, "_frozen", True)
        return

    def __repr__(self) -> str:
        """ """
        node = (
            f"{self.__class__.__name__}"
            f"(processor={self._processor!r}, defaults={self._defaults!r})"
        )
        return node

    @property
    def processor(self) -> Processor:
        """ """
        return self._processor

    @property
    def defaults(self) -> tuple[float, ...]:
        """ """
        return self._defaults

    @property
    def rbounds(self) -> tuple[tuple[float, float], ...]:
        """ """
        return self._processor.rbounds

    @property
    def tbounds(self) -> tuple[tuple[float, float], ...]:
        """ """
        return self._processor.tbounds

    @property
    def rnd(self) -> int:
        """ """
        return self._processor.rnd

    @property
    def tnd(self) -> int:
        """ """
        return self._processor.tnd

    def __eq__(self, other: object) -> bool:
        """ """
        # other validations
        if type(self) is not type(other):
            return NotImplemented
        # equality check
        is_equal = (
            self._processor == other._processor and self._defaults == other._defaults
        )
        return is_equal

    def copy(self) -> Node:
        """ """
        node = deepcopy(self)
        return node

    def to_dict(self) -> dict[str, Any]:
        """ """
        node = {
            "type": self.__class__.__name__,
            "processor": self._processor.to_dict(),
            "defaults": self._defaults,
        }
        return node

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Node:
        """ """
        # data validations
        if not isinstance(data, dict):
            raise TypeError("data must be a dict.")
        if "type" not in data:
            raise KeyError("data must contain the key 'type'.")
        if data["type"] != cls.__name__:
            raise ValueError(f"data['type'] must be a {cls.__name__}.")
        if "processor" not in data:
            raise KeyError("data must contain the key 'processor'.")
        if "defaults" not in data:
            raise KeyError("data must contain the key 'defaults'.")
        # initializations
        node = cls(
            Processor.from_dict(data["processor"]),
            data["defaults"],
        )
        return node

    def save(self, path: str | Path, overwrite: bool = False) -> None:
        """ """
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
    def load(cls, path: str | Path) -> Node:
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
        node = cls.from_dict(data)
        return node

    def __setattr__(self, name: str, value: Any) -> None:
        """ """
        # freeze check
        if getattr(self, "_frozen", False):
            raise AttributeError(f"{self.__class__.__name__} is immutable.")
        # set the attribute
        object.__setattr__(self, name, value)
        return

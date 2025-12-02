from __future__ import annotations

import json
from collections.abc import Iterator

import numpy as np

from jacinta.nodes import Cell, Processor, Receiver, Transmitter


class Jacinta:

    def __init__(
        self,
        size: tuple[int, int],
        receiver_params: dict[str, any] | None = None,
        processor_params: dict[str, any] | None = None,
        transmitter_params: dict[str, any] | None = None,
        cell_params: dict[str, any] | None = None,
    ) -> None:
        """ """
        receiver_kwargs = {"size": size[0]}
        if receiver_params:
            receiver_kwargs.update(receiver_params)
        self.receiver = Receiver(**receiver_kwargs)

        transmitter_kwargs = {"size": size[1]}
        if transmitter_params:
            transmitter_kwargs.update(transmitter_params)
        self.transmitter = Transmitter(**transmitter_kwargs)

        processor_kwargs = {"size": size[1]}
        if processor_params:
            processor_kwargs.update(processor_params)
        processor = Processor(**processor_kwargs)

        bounds = np.zeros((size[0], 2), dtype=float)
        bounds[:, 0] = -np.inf
        bounds[:, 1] = +np.inf
        cell_kwargs = {"bounds": bounds, "processor": processor}
        if cell_params:
            cell_kwargs.update(cell_params)
        self.root = Cell(**cell_kwargs)
        return

    @property
    def R(self) -> int:
        """ """
        R = self.receiver.N
        return R

    @property
    def T(self) -> int:
        """ """
        T = self.transmitter.N
        return T

    def copy(self) -> Jacinta:
        """ """
        jacinta = self.__class__.__new__(self.__class__)
        jacinta.receiver = self.receiver.copy()
        jacinta.transmitter = self.transmitter.copy()
        jacinta.root = self.root.copy()
        return jacinta

    def to_dict(self) -> dict[str, any]:
        """ """
        data = {
            "class": self.__class__.__name__,
            "receiver": self.receiver.to_dict(),
            "transmitter": self.transmitter.to_dict(),
            "root": self.root.to_dict(),
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, any]) -> Jacinta:
        """ """
        receiver = Receiver.from_dict(data["receiver"])
        transmitter = Transmitter.from_dict(data["transmitter"])
        root = Cell.from_dict(data["root"])
        jacinta = cls.__new__(cls)
        jacinta.receiver = receiver
        jacinta.transmitter = transmitter
        jacinta.root = root
        return jacinta

    def save(self, file_path: str) -> None:
        """ """
        data = self.to_dict()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return

    @classmethod
    def load(cls, file_path: str) -> Jacinta:
        """ """
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        jacinta = cls.from_dict(data)
        return jacinta

    def process_forward(self, x: np.ndarray, hit: bool = True) -> np.ndarray:
        """ """
        r = self.receiver.process_forward(x)
        p = self.root.process_forward(r, hit)
        t = self.transmitter.process_forward(p)
        return t

    def process_backward(self, x: np.ndarray, y: np.ndarray, r: float) -> None:
        """ """
        x = self.receiver.process_forward(x)
        y = self.transmitter.process_backward(y)
        self.root.process_backward(x, y, r)
        return

    def add_receiver(
        self, min_x: float | None = None, max_x: float | None = None
    ) -> None:
        """ """
        self.receiver.add_dimension(min_x, max_x)
        self.root.add_dimension(-np.inf, +np.inf)
        return

    def remove_receiver(self, idx: int) -> None:
        """ """
        self.receiver.remove_dimension(idx)
        self.root.remove_dimension(idx)
        return

    def add_transmitter(
        self, min_x: float | None = None, max_x: float | None = None
    ) -> None:
        """ """
        self.transmitter.add_dimension(min_x, max_x)
        for cell in self._iter_cells(self.root):
            cell.processor.add_dimension()
        return

    def remove_transmitter(self, idx: int) -> None:
        """ """
        self.transmitter.remove_dimension(idx)
        for cell in self._iter_cells(self.root):
            cell.processor.remove_dimension(idx)
        return

    def _iter_cells(self, cell: Cell) -> Iterator[Cell]:
        """ """
        cells = [cell]
        while cells:
            cell = cells.pop()
            yield cell
            for subcell in cell.subcells:
                cells.append(subcell)
        return

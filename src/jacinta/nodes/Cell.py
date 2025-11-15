from __future__ import annotations

import json

import numpy as np

from jacinta.nodes import Processor


class Cell:

    def __init__(
        self,
        bounds: np.ndarray,
        processor: Processor,
        lev: int = 1,
        max_lev: int = -1,
        lev_k: float = 1.1,
        min_lr: float = 1e-3,
        lr_k: float = 1e-3,
    ) -> None:
        """ """
        self.bounds = np.asarray(bounds, dtype=float).copy()
        self.min_x: np.ndarray = np.full(self.N, np.inf, dtype=float)
        self.max_x: np.ndarray = np.full(self.N, -np.inf, dtype=float)
        self.mids: np.ndarray = np.full(self.N, np.nan, dtype=float)

        self.lev = lev
        self.lev_k = lev_k
        self.max_lev = max_lev
        self.subcells: list[Cell | None] = [None] * (2**self.N)
        self.sub_res: np.ndarray = np.full(2**self.N, lev**lev_k, dtype=float)

        self.lr_k = lr_k
        self.min_lr = min_lr
        self.processor = processor.copy() if processor is not None else processor
        if self.processor is not None:
            self.processor.lr_mu = min_lr + (1.0 - min_lr) / (lr_k * lev + 1.0)
            self.processor.lr_sigma = self.processor.lr_mu * 0.1
        return

    @property
    def N(self) -> int:
        """ """
        N = self.bounds.shape[0]
        return N

    @property
    def L(self) -> int:
        """ """
        L = 1
        subcells = [subcell for subcell in self.subcells if subcell is not None]
        if subcells:
            L += max(subcell.L for subcell in subcells)
        return L

    def copy(self) -> Cell:
        """ """
        cell = self.__class__(
            self.bounds,
            self.processor,
            self.lev,
            self.max_lev,
            self.lev_k,
            self.min_lr,
            self.lr_k,
        )
        cell.min_x = self.min_x.copy()
        cell.max_x = self.max_x.copy()
        cell.mids = self.mids.copy()
        cell.sub_res = self.sub_res.copy()
        cell.subcells = [
            subcell.copy() if subcell is not None else subcell
            for subcell in self.subcells
        ]
        return cell

    def to_dict(self) -> dict[str, any]:
        """ """
        data = {
            "class": self.__class__.__name__,
            "bounds": self.bounds.tolist(),
            "processor": (
                self.processor.to_dict()
                if self.processor is not None
                else self.processor
            ),
            "lev": self.lev,
            "max_lev": self.max_lev,
            "lev_k": self.lev_k,
            "sub_res": self.sub_res.tolist(),
            "min_lr": self.min_lr,
            "lr_k": self.lr_k,
            "subcells": [
                subcell.to_dict() if subcell is not None else subcell
                for subcell in self.subcells
            ],
            "min_x": self.min_x.tolist(),
            "max_x": self.max_x.tolist(),
            "mids": self.mids.tolist(),
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, any]) -> Cell:
        """ """
        cell = cls(
            bounds=np.array(data["bounds"], dtype=float),
            processor=(
                Processor.from_dict(data["processor"])
                if data["processor"] is not None
                else data["processor"]
            ),
            lev=int(data["lev"]),
            max_lev=int(data["max_lev"]),
            lev_k=float(data["lev_k"]),
            min_lr=float(data["min_lr"]),
            lr_k=float(data["lr_k"]),
        )
        cell.sub_res = np.asarray(data["sub_res"], dtype=float).copy()
        cell.min_x = np.asarray(data["min_x"], dtype=float).copy()
        cell.max_x = np.asarray(data["max_x"], dtype=float).copy()
        cell.mids = np.asarray(data["mids"], dtype=float).copy()
        cell.subcells = [
            cls.from_dict(subcell) if subcell is not None else subcell
            for subcell in data["subcells"]
        ]
        return cell

    def save(self, file_path: str) -> None:
        """ """
        data = self.to_dict()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return

    @classmethod
    def load(cls, file_path: str) -> Cell:
        """ """
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        cell = cls.from_dict(data)
        return cell

    def contains(self, x: np.ndarray) -> bool:
        """ """
        contains = np.all((x >= self.bounds[:, 0]) & (x < self.bounds[:, 1]))
        return contains

    def find(self, x: np.ndarray) -> Cell | None:
        """ """
        cell = None
        if self.contains(x):
            for subcell in self.subcells:
                if subcell and subcell.contains(x):
                    cell = subcell.find(x)
                    if cell is not None:
                        break
            cell = cell if cell is not None else self
        return cell

    def hit(self, x: np.ndarray) -> None:
        """ """
        cell = self.find(x)
        if cell is not None:
            cell.min_x = np.minimum(cell.min_x, x)
            cell.max_x = np.maximum(cell.max_x, x)
            if cell.max_lev < 0 or cell.lev < cell.max_lev:
                sub_idx = cell._get_subcell_index(x)
                cell.sub_res[sub_idx] -= 1.0
                if cell.sub_res[sub_idx] <= 0.0:
                    cell._create_subcell(sub_idx)
        return

    def add_dimension(self, low: float, high: float) -> None:
        """ """
        new_bounds = np.array([[low, high]], dtype=float)
        self.bounds = np.concatenate([self.bounds, new_bounds])

        new_min_x = np.array([np.inf], dtype=float)
        new_max_x = np.array([-np.inf], dtype=float)

        self.min_x = np.concatenate([self.min_x, new_min_x])
        self.max_x = np.concatenate([self.max_x, new_max_x])

        new_mid = np.array([np.nan], dtype=float)
        self.mids = np.concatenate([self.mids, new_mid])

        new_subcells = [None] * (2**self.N)
        new_sub_res = np.empty(2**self.N, dtype=float)

        for idx, subcell in enumerate(self.subcells):
            idx0 = idx
            idx1 = idx | (1 << (self.N - 1))
            new_subcells[idx0] = subcell
            new_subcells[idx1] = subcell
            if subcell is not None:
                subcell.add_dimension(low, high)

            res = self.sub_res[idx]
            new_sub_res[idx0] = res
            new_sub_res[idx1] = res

        self.subcells = new_subcells
        self.sub_res = new_sub_res
        return

    def remove_dimension(self, idx: int) -> None:
        """ """
        raise NotImplementedError

    def _update_mids(self):
        """ """
        idx = np.where(np.isnan(self.mids))[0]
        mids = np.mean(self.bounds[idx], axis=1)
        for k, d in enumerate(idx):
            low, high = self.bounds[d]
            if not np.isfinite(mids[k]):
                min_x = self.min_x[d]
                max_x = self.max_x[d]
                if np.isneginf(low) and np.isposinf(high):
                    mids[k] = 0.0
                elif np.isneginf(low) and np.isfinite(high):
                    mids[k] = min_x - 1.0
                elif np.isfinite(low) and np.isposinf(high):
                    mids[k] = max_x + 1.0
        self.mids[idx] = mids
        return

    def _get_subcell_index(self, x: np.ndarray) -> int:
        """ """
        idx = 0
        self._update_mids()
        for d in range(self.N):
            if x[d] > self.mids[d]:
                idx |= 1 << d
        return idx

    def _create_subcell(self, idx: int) -> None:
        """ """
        self._update_mids()
        subcell_bounds = np.zeros_like(self.bounds)
        for d in range(self.N):
            low, high = self.bounds[d]
            mid = self.mids[d]
            take_upper = (idx >> d) & 1
            if take_upper == 0:
                subcell_bounds[d, 0] = low
                subcell_bounds[d, 1] = mid
            else:
                subcell_bounds[d, 0] = mid
                subcell_bounds[d, 1] = high

        subcell = self.__class__(
            bounds=subcell_bounds,
            processor=self.processor,
            lev=self.lev + 1,
            max_lev=self.max_lev,
            lev_k=self.lev_k,
            min_lr=self.min_lr,
            lr_k=self.lr_k,
        )
        self.subcells[idx] = subcell
        return

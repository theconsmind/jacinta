from __future__ import annotations

import json

import numpy as np

from jacinta.nodes import Processor


class Cell:
    """ """

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
        self.childs: list[Cell | None] = [None] * (2**self.N)
        self.res: np.ndarray = np.full(2**self.N, lev**lev_k, dtype=float)
        self.parent: Cell | None = None

        self.lr_k = lr_k
        self.min_lr = min_lr
        self.processor = processor.copy()
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
        childs = [child for child in self.childs if child]
        if childs:
            L += max(child.L for child in childs)
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
        cell.res = self.res.copy()
        cell.childs = [
            child.copy() if child else child
            for child in self.childs
        ]

        for child in cell.childs:
            if child:
                child.parent = cell
        return cell

    def to_dict(self) -> dict[str, any]:
        """ """
        data = {
            "class": self.__class__.__name__,
            "bounds": self.bounds.tolist(),
            "processor": (
                self.processor.to_dict()
                if self.processor
                else self.processor
            ),
            "lev": self.lev,
            "max_lev": self.max_lev,
            "lev_k": self.lev_k,
            "min_lr": self.min_lr,
            "lr_k": self.lr_k,
            "min_x": self.min_x.tolist(),
            "max_x": self.max_x.tolist(),
            "mids": self.mids.tolist(),
            "res": self.res.tolist(),
            "childs": [
                child.to_dict() if child else child
                for child in self.childs
            ],
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, any]) -> Cell:
        """ """
        cell = cls(
            bounds=np.array(data["bounds"], dtype=float),
            processor=(
                Processor.from_dict(data["processor"])
                if data["processor"]
                else data["processor"]
            ),
            lev=int(data["lev"]),
            max_lev=int(data["max_lev"]),
            lev_k=float(data["lev_k"]),
            min_lr=float(data["min_lr"]),
            lr_k=float(data["lr_k"]),
        )
        cell.min_x = np.asarray(data["min_x"], dtype=float).copy()
        cell.max_x = np.asarray(data["max_x"], dtype=float).copy()
        cell.mids = np.asarray(data["mids"], dtype=float).copy()
        cell.res = np.asarray(data["res"], dtype=float).copy()
        cell.childs = [
            cls.from_dict(child) if child else child
            for child in data["childs"]
        ]

        for child in cell.childs:
            if child:
                child.parent = cell
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

    def process_forward(self, x: np.ndarray, hit: bool = True) -> np.ndarray | None:
        """ """
        cell = self._hit(x, hit=hit)
        y = cell.processor.process_forward() if cell else None
        return y

    def process_backward(self, x: np.ndarray, y: np.ndarray, r: float) -> None:
        """ """
        cell = self._hit(x, hit=False)
        while cell:
            cell.processor.process_backward(y, r)
            cell = cell.parent
        return

    def _contains(self, x: np.ndarray) -> bool:
        """ """
        contains = np.all((x >= self.bounds[:, 0]) & (x < self.bounds[:, 1]))
        return contains

    def _find(self, x: np.ndarray) -> Cell | None:
        """ """
        cell = None
        if self._contains(x):
            cell = self
            for child in self.childs:
                if child and child._contains(x):
                    cell = child._find(x)
                    break
        return cell

    def _update_mids(self) -> None:
        """ """
        idx = np.where(np.isnan(self.mids))[0]
        mids = np.mean(self.bounds, axis=1)
        for dim in idx:
            if not np.isfinite(mids[dim]):
                low, high = self.bounds[dim]
                min_x = self.min_x[dim]
                max_x = self.max_x[dim]
                if np.isneginf(low) and np.isposinf(high):
                    mids[dim] = 0.0
                elif np.isneginf(low) and np.isfinite(high):
                    mids[dim] = min_x - 1e-12
                elif np.isfinite(low) and np.isposinf(high):
                    mids[dim] = max_x + 1e-12
        self.mids[idx] = mids[idx]
        return

    def _get_child_idx(self, x: np.ndarray) -> int:
        """ """
        idx = 0
        self._update_mids()
        for dim in range(self.N):
            if x[dim] >= self.mids[dim]:
                idx |= 1 << dim
        return idx

    def _create_child(self, idx: int) -> Cell:
        """ """
        child_bounds = np.zeros_like(self.bounds)
        for dim in range(self.N):
            low, high = self.bounds[dim]
            mid = self.mids[dim]
            take_upper = (idx >> dim) & 1
            if take_upper == 0:
                child_bounds[dim, 0] = low
                child_bounds[dim, 1] = mid
            else:
                child_bounds[dim, 0] = mid
                child_bounds[dim, 1] = high

        child = self.__class__(
            bounds=child_bounds,
            processor=self.processor,
            lev=self.lev + 1,
            max_lev=self.max_lev,
            lev_k=self.lev_k,
            min_lr=self.min_lr,
            lr_k=self.lr_k,
        )
        self.childs[idx] = child
        child.parent = self
        return child

    def _hit(self, x: np.ndarray, hit: bool = True) -> Cell | None:
        """ """
        cell = self._find(x)
        if cell:
            cell.min_x = np.minimum(cell.min_x, x)
            cell.max_x = np.maximum(cell.max_x, x)
            if hit and (cell.max_lev < 0 or cell.lev < cell.max_lev):
                child_idx = cell._get_child_idx(x)
                cell.res[child_idx] -= 1.0

                mirror_idx = child_idx ^ (1 << (cell.N - 1))
                if cell.childs[mirror_idx] is cell.childs[child_idx]:
                    cell.res[mirror_idx] = cell.res[child_idx]

                if cell.res[child_idx] <= 0.0:
                    cell = cell._create_child(child_idx)
        return cell

    def add_dimension(
        self, low: float, high: float, mu: float = 0.0, sigma: float = 100.0
    ) -> None:
        """ """
        new_bounds = np.array([[low, high]], dtype=float)
        self.bounds = np.concatenate([self.bounds, new_bounds], axis=0)

        new_min_x = np.array([np.inf], dtype=float)
        new_max_x = np.array([-np.inf], dtype=float)

        self.min_x = np.concatenate([self.min_x, new_min_x])
        self.max_x = np.concatenate([self.max_x, new_max_x])

        new_mid = np.array([np.nan], dtype=float)
        self.mids = np.concatenate([self.mids, new_mid])

        self.processor.add_dimension(mu, sigma)

        new_childs = [None] * (2**self.N)
        new_res = np.empty(2**self.N, dtype=float)

        for idx, child in enumerate(self.childs):
            idx0 = idx
            idx1 = idx | (1 << (self.N - 1))
            new_childs[idx0] = child
            new_childs[idx1] = child
            if child:
                child.add_dimension(low, high)

            res = self.res[idx]
            new_res[idx0] = res
            new_res[idx1] = res

        self.childs = new_childs
        self.res = new_res
        return

    def remove_dimension(self, idx: int) -> None:
        """ """
        # self.bounds = np.delete(self.bounds, idx, axis=0)
        # self.min_x = np.delete(self.min_x, idx)
        # self.max_x = np.delete(self.max_x, idx)
        # self.mids = np.delete(self.mids, idx)
        # self.processor.remove_dimension(idx)
        #
        # new_childs = [None] * (2**self.N)
        # new_res = np.empty(2**self.N, dtype=float)
        #
        # for idx, child in enumerate(self.childs):
        #     child.remove_dimension(idx)
        raise NotImplementedError

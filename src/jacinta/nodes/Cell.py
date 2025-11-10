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
        """
        Initialize a spatial Cell with recursive subdivision capability.

        Args:
            bounds (np.ndarray): Array of shape (N, 2) with [low, high] per dimension.
            processor (Processor): Payload to be stored in the Cell.
            lev (int): Current subdivision level (root is 0).
            max_lev (int): Maximum allowed subdivision level.
            lev_k (float): Coefficient used in the Subcells resolution formula.
            min_lr (float): Minimum learning rate in the lr schedule.
            lr_k (float): Coefficient used in the lr schedule.

        Returns:
            None
        """
        self.bounds = bounds.astype(float).copy()
        self.subcells: list[Cell] = list()
        self.processor = processor.copy()
        self.max_lev = max_lev
        self.lev = lev
        self.lev_k = lev_k
        self.res: int = lev**lev_k
        self.base_res: int = self.res
        self.min_lr = min_lr
        self.lr_k = lr_k
        self.processor.lr_mu = min_lr + (1.0 - min_lr) / (lr_k * lev + 1.0)
        self.processor.lr_sigma = self.processor.lr_mu * 0.1
        self.min_x: np.ndarray = np.full(self.bounds.shape[0], np.inf, dtype=float)
        self.max_x: np.ndarray = np.full(self.bounds.shape[0], -np.inf, dtype=float)
        return

    @property
    def N(self) -> int:
        """
        Return the dimensionality of the Cell.

        Args:
            None

        Returns:
            int: Number of dimensions of the bounds.
        """
        N = self.bounds.shape[0]
        return N

    @property
    def L(self) -> int:
        """
        Return the maximum depth of the subtree rooted at this Cell.

        Args:
            None

        Returns:
            int: Tree depth starting from this node.
        """
        L = 1
        if self.subcells:
            L += max(subcell.L for subcell in self.subcells)
        return L

    def copy(self) -> Cell:
        """
        Create a deep copy of this Cell, including its subtree.

        Args:
            None

        Returns:
            Cell: A new Cell with the same configuration and Subcells.
        """
        cell = self.__class__(
            self.bounds,
            self.processor,
            self.lev,
            self.max_lev,
            self.lev_k,
            self.min_lr,
            self.lr_k,
        )
        cell.res = self.res
        cell.subcells = [subcell.copy() for subcell in self.subcells]
        cell.min_x = self.min_x.copy()
        cell.max_x = self.max_x.copy()
        return cell

    def to_dict(self) -> dict[str, any]:
        """
        Serialize Cell and its subtree to a dictionary.

        Args:
            None

        Returns:
            dict[str, any]: Serializable snapshot of the Cell.
        """
        data = {
            "class": self.__class__.__name__,
            "bounds": self.bounds.tolist(),
            "processor": self.processor.to_dict(),
            "lev": self.lev,
            "max_lev": self.max_lev,
            "lev_k": self.lev_k,
            "res": self.res,
            "base_res": self.base_res,
            "min_lr": self.min_lr,
            "lr_k": self.lr_k,
            "subcells": [subcell.to_dict() for subcell in self.subcells],
            "min_x": self.min_x.tolist(),
            "max_x": self.max_x.tolist(),
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, any]) -> Cell:
        """
        Internal helper to reconstruct a Cell (and its subtree) from a dict.

        Args:
            data (dict[str, any]): Serialized Cell data.

        Returns:
            Cell: Reconstructed Cell instance.
        """
        cell = cls(
            bounds=np.array(data["bounds"], dtype=float),
            processor=Processor.from_dict(data["processor"]),
            lev=int(data["lev"]),
            max_lev=int(data["max_lev"]),
            lev_k=float(data["lev_k"]),
            min_lr=float(data["min_lr"]),
            lr_k=float(data["lr_k"]),
        )
        cell.res = int(data["res"])
        cell.subcells = [cls.from_dict(subcell) for subcell in data["subcells"]]
        cell.min_x = np.array(data["min_x"], dtype=float)
        cell.max_x = np.array(data["max_x"], dtype=float)
        return cell

    def save(self, file_path: str) -> None:
        """
        Save Cell tree as JSON to disk.

        Args:
            file_path (str): Target path.

        Returns:
            None
        """
        data = self.to_dict()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return

    @classmethod
    def load(cls, file_path: str) -> Cell:
        """
        Load Cell tree from a JSON file.

        Args:
            file_path (str): Source JSON path.

        Returns:
            Cell: Reconstructed Cell instance.
        """
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        cell = cls.from_dict(data)
        return cell

    def contains(self, x: np.ndarray) -> bool:
        """
        Check whether a point lies inside the Cell bounds.

        Args:
            x (np.ndarray): Point coordinates of shape (N,).

        Returns:
            bool: True if the point is inside, False otherwise.
        """
        contains = np.all((x >= self.bounds[:, 0]) & (x <= self.bounds[:, 1]))
        return contains

    def find(self, x: np.ndarray) -> Cell | None:
        """
        Recursively find the deepest Subcell that contains x.

        Args:
            x (np.ndarray): Query point.

        Returns:
            Cell | None: Deepest Cell containing x, or None if x is outside.
        """
        cell = None
        if self.contains(x):
            for subcell in self.subcells:
                if subcell.contains(x):
                    cell = subcell.find(x)
                    if cell is not None:
                        break
            cell = cell if cell is not None else self
        return cell

    def hit(self, x: np.ndarray) -> None:
        """
        Register a hit at the Cell containing x.
        When the hit budget reaches zero, the Cell is split.

        Args:
            x (np.ndarray): Point where the hit occurred.

        Returns:
            None
        """
        cell = self.find(x)
        if cell is not None and cell.res > 0:
            cell.min_x = np.minimum(cell.min_x, x)
            cell.max_x = np.maximum(cell.max_x, x)
            cell.res -= 1
            if cell.res <= 0:
                cell._split()
        return

    def _split(self) -> None:
        """
        Split current Cell into 2^N Subcells if level allows it.

        Args:
            x (np.ndarray): Point where the split occurred.

        Returns:
            None
        """
        if self.max_lev < 0 or self.lev < self.max_lev:
            mids = np.mean(self.bounds, axis=1)
            for d in range(self.N):
                low, high = self.bounds[d]
                if not np.isfinite(mids[d]):
                    min_x = self.min_x[d]
                    max_x = self.max_x[d]
                    if np.isneginf(low) and np.isposinf(high):
                        mids[d] = 0.0
                    elif np.isneginf(low) and np.isfinite(high):
                        mids[d] = abs(min_x)
                    elif np.isfinite(low) and np.isposinf(high):
                        mids[d] = abs(max_x)
            subcells = list()
            for mask in range(2**self.N):
                subcell_bounds = np.zeros_like(self.bounds)
                for d in range(self.N):
                    low, high = self.bounds[d]
                    mid = mids[d]
                    take_upper = (mask >> d) & 1
                    if take_upper == 0:
                        subcell_bounds[d, 0] = low
                        subcell_bounds[d, 1] = mid
                    else:
                        subcell_bounds[d, 0] = mid
                        subcell_bounds[d, 1] = high
                subcell = self.__class__(
                    subcell_bounds,
                    self.processor,
                    self.lev + 1,
                    self.max_lev,
                    self.lev_k,
                    self.min_lr,
                    self.lr_k,
                )
                subcells.append(subcell)
            self.subcells = subcells
        return

    def add_dimension(self, low: float, high: float) -> None:
        """
        Add a new dimension to this Cell and to all its Subcells.

        Args:
            low (float): Lower bound of the new dimension.
            high (float): Upper bound of the new dimension.

        Returns:
            None
        """
        new_row = np.array([[low, high]], dtype=float)
        self.bounds = np.concatenate([self.bounds, new_row], axis=0)
        self.min_x = np.concatenate([self.min_x, np.array([np.inf], dtype=float)])
        self.max_x = np.concatenate([self.max_x, np.array([-np.inf], dtype=float)])
        for subcell in self.subcells:
            subcell.add_dimension(low, high)
        return

    def remove_dimension(self, idx: int) -> None:
        """
        Remove a dimension from this Cell and all of its Subcells.

        Args:
            idx (int): Index of the dimension to remove.

        Returns:
            None
        """
        self.bounds = np.delete(self.bounds, idx, axis=0)
        self.min_x = np.delete(self.min_x, idx, axis=0)
        self.max_x = np.delete(self.max_x, idx, axis=0)
        for subcell in self.subcells:
            subcell.remove_dimension(idx)
        return

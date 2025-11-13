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
            lev (int): Current subdivision level (root is 1).
            max_lev (int): Maximum allowed subdivision level.
            lev_k (float): Coefficient used in the Subcells resolution formula.
            min_lr (float): Minimum learning rate in the lr schedule.
            lr_k (float): Coefficient used in the lr schedule.
        """
        self.bounds = np.asarray(bounds, dtype=float).copy()
        self.min_x: np.ndarray = np.full(self.N, np.inf, dtype=float)
        self.max_x: np.ndarray = np.full(self.N, -np.inf, dtype=float)
        self._mids: np.ndarray | None = None

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
        """
        Return the dimensionality of the Cell.

        Returns:
            int: Number of dimensions of the bounds.
        """
        N = self.bounds.shape[0]
        return N

    @property
    def L(self) -> int:
        """
        Return the maximum depth of the subtree rooted at this Cell.

        Returns:
            int: Tree depth starting from this node.
        """
        L = 1
        subcells = [subcell for subcell in self.subcells if subcell is not None]
        if subcells:
            L += max(subcell.L for subcell in subcells)
        return L

    def copy(self) -> Cell:
        """
        Create a deep copy of this Cell, including its subtree.

        Returns:
            Cell: A new Cell with the same configuration and subcells.
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
        cell.min_x = self.min_x.copy()
        cell.max_x = self.max_x.copy()
        cell._mids = self._mids.copy() if self._mids is not None else self._mids
        cell.sub_res = self.sub_res.copy()
        cell.subcells = [
            subcell.copy() if subcell is not None else subcell
            for subcell in self.subcells
        ]
        return cell

    def to_dict(self) -> dict[str, any]:
        """
        Serialize Cell and its subtree to a dictionary.

        Returns:
            dict[str, any]: Serializable snapshot of the Cell.
        """
        data = {
            "class": self.__class__.__name__,
            "bounds": self.bounds.tolist(),
            "processor": self.processor.to_dict() if self.processor is not None else self.processor,
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
            "mids": self._mids.tolist() if self._mids is not None else self._mids,
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, any]) -> Cell:
        """
        Reconstruct a Cell (and its subtree) from a dict.

        Args:
            data (dict[str, any]): Serialized Cell data.

        Returns:
            Cell: Reconstructed Cell instance.
        """
        cell = cls(
            bounds=np.array(data["bounds"], dtype=float),
            processor=Processor.from_dict(data["processor"]) if data["processor"] is not None else data["processor"],
            lev=int(data["lev"]),
            max_lev=int(data["max_lev"]),
            lev_k=float(data["lev_k"]),
            min_lr=float(data["min_lr"]),
            lr_k=float(data["lr_k"]),
        )
        cell.sub_res = np.asarray(data["sub_res"], dtype=float).copy()
        cell.min_x = np.asarray(data["min_x"], dtype=float).copy()
        cell.max_x = np.asarray(data["max_x"], dtype=float).copy()
        cell._mids = np.asarray(data["mids"], dtype=float).copy() if data["mids"] is not None else data["mids"]
        cell.subcells = [
            cls.from_dict(subcell) if subcell is not None else subcell
            for subcell in data["subcells"]
        ]
        return cell

    def save(self, file_path: str) -> None:
        """
        Save Cell tree as JSON to disk.

        Args:
            file_path (str): Target path.
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
        contains = np.all((x >= self.bounds[:, 0]) & (x < self.bounds[:, 1]))
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
                if subcell and subcell.contains(x):
                    cell = subcell.find(x)
                    if cell is not None:
                        break
            cell = cell if cell is not None else self
        return cell

    def hit(self, x: np.ndarray) -> None:
        """
        Register a hit at the leaf Cell containing x.
        When the hit budget for the corresponding subcell reaches zero,
        that subcell is materialized as a new Cell.

        Args:
            x (np.ndarray): Point where the hit occurred.
        """
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

    #def add_dimension(self, low: float, high: float) -> None:
    #    """
    #    Add a new dimension to this Cell and to all its Subcells.
    #
    #    Args:
    #        low (float): Lower bound of the new dimension.
    #        high (float): Upper bound of the new dimension.
    #    """
    #    new_row = np.array([[low, high]], dtype=float)
    #    self.bounds = np.concatenate([self.bounds, new_row], axis=0)
    #    self.min_x = np.concatenate([self.min_x, np.array([np.inf], dtype=float)])
    #    self.max_x = np.concatenate([self.max_x, np.array([-np.inf], dtype=float)])
    #    for subcell in self.subcells:
    #        subcell.add_dimension(low, high)
    #    return
    #
    #def remove_dimension(self, idx: int) -> None:
    #    """
    #    Remove a dimension from this Cell and all of its Subcells.
    #
    #    Args:
    #        idx (int): Index of the dimension to remove.
    #    """
    #    self.bounds = np.delete(self.bounds, idx, axis=0)
    #    self.min_x = np.delete(self.min_x, idx, axis=0)
    #    self.max_x = np.delete(self.max_x, idx, axis=0)
    #    for subcell in self.subcells:
    #        subcell.remove_dimension(idx)
    #    return

    def _compute_mids(self) -> np.ndarray:
        """
        TODO
        """
        mids = np.mean(self.bounds, axis=1)
        for d in range(self.N):
            low, high = self.bounds[d]
            if not np.isfinite(mids[d]):
                min_x = self.min_x[d]
                max_x = self.max_x[d]
                if np.isneginf(low) and np.isposinf(high):
                    mids[d] = 0.0
                elif np.isneginf(low) and np.isfinite(high):
                    mids[d] = min_x - 1.0
                elif np.isfinite(low) and np.isposinf(high):
                    mids[d] = max_x + 1.0
        return mids

    def _get_subcell_index(self, x: np.ndarray) -> int:
        """
        TODO
        """
        idx = 0
        self._mids = self._mids if self._mids is not None else self._compute_mids()
        for d in range(self.N):
            if x[d] > self._mids[d]:
                idx |= (1 << d)
        return idx

    def _create_subcell(self, idx: int) -> None:
        """
        # TODO
        """
        self._mids = self._mids if self._mids is not None else self._compute_mids()
        subcell_bounds = np.zeros_like(self.bounds)
        for d in range(self.N):
            low, high = self.bounds[d]
            mid = self._mids[d]
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

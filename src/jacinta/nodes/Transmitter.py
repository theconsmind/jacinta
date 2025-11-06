from __future__ import annotations

import json

import numpy as np


class Transmitter:

    def __init__(
        self,
        size: int,
        min_x: float | np.ndarray | None = None,
        max_x: float | np.ndarray | None = None,
    ) -> None:
        """
        Initialize a Transmitter that maps unbounded signals to bounded ranges.

        Args:
            size (int): Dimensionality of the signal.
            min_x (float | np.ndarray | None): Lower bounds per dimension.
                np.nan means "no lower bound".
            max_x (float | np.ndarray | None): Upper bounds per dimension.
                np.nan means "no upper bound".

        Returns:
            None
        """
        if min_x is None:
            self.min_x = np.full(size, np.nan, dtype=float)
        elif isinstance(min_x, float):
            self.min_x = np.full(size, min_x, dtype=float)
        else:
            self.min_x = min_x.astype(float).copy()
        if max_x is None:
            self.max_x = np.full(size, np.nan, dtype=float)
        elif isinstance(max_x, float):
            self.max_x = np.full(size, max_x, dtype=float)
        else:
            self.max_x = np.asarray(max_x, dtype=float)
        self.has_bounds: np.ndarray = np.isfinite(self.min_x) & np.isfinite(self.max_x)
        return

    @property
    def N(self) -> int:
        """
        Return the dimensionality of the Transmitter.

        Args:
            None

        Returns:
            int: Number of dimensions.
        """
        N = self.min_x.size
        return N

    def copy(self) -> Transmitter:
        """
        Create a deep copy of the current Transmitter.

        Args:
            None

        Returns:
            Transmitter: A new Transmitter with the same parameters.
        """
        transmitter = self.__class__(self.N, self.min_x, self.max_x)
        return transmitter

    def to_dict(self) -> dict[str, any]:
        """
        Serialize Transmitter configuration to a dictionary.

        Args:
            None

        Returns:
            dict[str, any]: Serializable snapshot of the Transmitter.
        """
        data = {
            "class": self.__class__.__name__,
            "size": self.N,
            "min_x": self.min_x.tolist(),
            "max_x": self.max_x.tolist(),
            "has_bounds": self.has_bounds.tolist(),
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, any]) -> Transmitter:
        """
        Build a Transmitter instance from a dict.

        Args:
            data (dict[str, any]): Serialized Transmitter config.

        Returns:
            Transmitter: Reconstructed Transmitter instance.
        """
        transmitter = cls(
            size=int(data["size"]),
            min_x=np.array(data["min_x"], dtype=float),
            max_x=np.array(data["max_x"], dtype=float),
        )
        return transmitter

    def save(self, file_path: str) -> None:
        """
        Save transmitter configuration as JSON to given file.

        Args:
            file_path (str): Target file path.

        Returns:
            None
        """
        data = self.to_dict()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        return

    @classmethod
    def load(cls, file_path: str) -> Transmitter:
        """
        Load a Transmitter instance from a JSON file.

        Args:
            file_path (str): Path to the JSON file containing saved configuration.

        Returns:
            Transmitter: Reconstructed Transmitter instance.
        """
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        transmitter = cls.from_dict(data)
        return transmitter

    def process_forward(self, x: np.ndarray) -> np.ndarray:
        """
        Map internal unbounded vector x to the bounded external space.

        Args:
            x (np.ndarray): Internal control values in ℝ.

        Returns:
            np.ndarray: External values clamped to [min_x, max_x] where applicable.
        """
        y = x.copy()
        idx = np.where(self.has_bounds)[0]
        if idx.size > 0:
            x = np.tanh(x[idx])
            x = (x + 1.0) * 0.5
            y[idx] = self.min_x[idx] + x * (self.max_x[idx] - self.min_x[idx])
        return y

    def add_dimension(
        self, min_x: float | None = None, max_x: float | None = None
    ) -> None:
        """
        Add a new dimension with optional bounds.

        Args:
            min_x (float | None): Lower bound for the new dimension.
                np.nan means "no lower bound".
            max_x (float | None): Upper bound for the new dimension.
                np.nan means "no upper bound".

        Returns:
            None
        """
        new_min_x = np.array([min_x], dtype=float)
        new_max_x = np.array([max_x], dtype=float)
        self.min_x = np.concatenate([self.min_x, new_min_x])
        self.max_x = np.concatenate([self.max_x, new_max_x])
        self.has_bounds = np.isfinite(self.min_x) & np.isfinite(self.max_x)
        return

    def remove_dimension(self, idx: int) -> None:
        """
        Remove one dimension and its bounds.

        Args:
            idx (int): Index of the dimension to remove.

        Returns:
            None
        """
        self.min_x = np.delete(self.min_x, idx, axis=0)
        self.max_x = np.delete(self.max_x, idx, axis=0)
        self.has_bounds = np.delete(self.has_bounds, idx, axis=0)
        return

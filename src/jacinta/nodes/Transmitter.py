from __future__ import annotations

import json

import numpy as np


class Transmitter:
    """
    Transmitter is responsible for mapping Jacinta's internal control
    values back into the external, possibly bounded space.

    It acts as the "inverse" counterpart of Receiver:
        - internal space:   typically unbounded (ℝ)
        - external space:   finite intervals [min_x[i], max_x[i]]

    Only fully bounded dimensions are rescaled back using a tanh-based
    squashing transform.
    """

    def __init__(
        self,
        size: int,
        min_x: float | np.ndarray | None = None,
        max_x: float | np.ndarray | None = None,
        eps: float = 1e-12,
    ) -> None:
        """
        Initialize a Transmitter that maps unbounded signals to bounded ranges.

        Args:
            size (int): Dimensionality of the signal.
            min_x (float | np.ndarray | None): Lower bounds per dimension.
                np.nan means "no lower bound".
            max_x (float | np.ndarray | None): Upper bounds per dimension.
                np.nan means "no upper bound".
            eps (float): Numerical epsilon for stability.
        """
        # Numerical safety margin used when approaching the boundaries
        # of the normalized interval (-1, 1). Prevents division by zero
        # and infinities in the log / artanh transform.
        self.eps = eps

        # Normalize lower bounds to a float array:
        # - None  -> all dimensions unbounded below    -> fill with NaN
        # - float -> same lower bound for all dims     -> broadcast scalar
        # - array -> per-dimension lower bounds        -> copy to own memory
        if min_x is None:
            self.min_x = np.full(size, np.nan, dtype=float)
        elif isinstance(min_x, float):
            self.min_x = np.full(size, min_x, dtype=float)
        else:
            self.min_x = np.asarray(min_x, dtype=float).copy()

        # Same normalization logic for upper bounds.
        if max_x is None:
            self.max_x = np.full(size, np.nan, dtype=float)
        elif isinstance(max_x, float):
            self.max_x = np.full(size, max_x, dtype=float)
        else:
            self.max_x = np.asarray(max_x, dtype=float)

        # Dimensions with both bounds finite are considered "fully bounded"
        # and will be transformed back into their [min_x, max_x] interval.
        self.has_bounds: np.ndarray = np.isfinite(self.min_x) & np.isfinite(self.max_x)
        return

    @property
    def N(self) -> int:
        """
        Return the dimensionality of the Transmitter.

        Returns:
            int: Number of dimensions.
        """
        # Dimensionality is given by the size of the bounds vectors.
        N = self.min_x.size
        return N

    def copy(self) -> Transmitter:
        """
        Create a deep copy of the current Transmitter.

        Returns:
            Transmitter: A new Transmitter with the same parameters.
        """
        # Reuse the constructor so future changes in __init__ are honored.
        transmitter = self.__class__(self.N, self.min_x, self.max_x, self.eps)
        return transmitter

    def to_dict(self) -> dict[str, any]:
        """
        Serialize Transmitter configuration to a dictionary.

        Returns:
            dict[str, any]: Serializable snapshot of the Transmitter.
        """
        # Convert numpy arrays into JSON-friendly Python lists.
        data = {
            "class": self.__class__.__name__,
            "size": self.N,
            "min_x": self.min_x.tolist(),
            "max_x": self.max_x.tolist(),
            "has_bounds": self.has_bounds.tolist(),
            "eps": self.eps,
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
        # Reconstruct core arrays and delegate initialization to __init__.
        transmitter = cls(
            size=int(data["size"]),
            min_x=np.array(data["min_x"], dtype=float),
            max_x=np.array(data["max_x"], dtype=float),
            eps=float(data["eps"]),
        )
        return transmitter

    def save(self, file_path: str) -> None:
        """
        Save transmitter configuration as JSON to given file.

        Args:
            file_path (str): Target file path.
        """
        # Persist configuration only (no runtime-dependent state).
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
        # Mirror the save() path: JSON -> dict -> Transmitter.
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
            np.ndarray: External values mapped into [min_x, max_x] where applicable.
        """
        # Work on a copy to avoid mutating the caller's array.
        y = x.copy()

        # Indices of dimensions that have both finite bounds.
        idx = np.where(self.has_bounds)[0]

        if idx.size > 0:
            # 1) Squash unbounded values in ℝ into (-1, 1) using tanh.
            x = np.tanh(x[idx])

            # 2) Linearly map (-1, 1) -> (0, 1).
            x = (x + 1.0) * 0.5

            # 3) Rescale (0, 1) into the original [min_x, max_x] interval.
            y[idx] = self.min_x[idx] + x * (self.max_x[idx] - self.min_x[idx])

        # Unbounded or partially bounded dimensions pass through unchanged.
        return y

    def process_backward(self, x: np.ndarray) -> np.ndarray:
        """
        Transform bounded components of x into unbounded space.

        Args:
            x (np.ndarray): Input vector in the original space.

        Returns:
            np.ndarray: Output vector where bounded dims are mapped to ℝ.
        """
        # Start from a copy to avoid mutating the caller's array.
        y = x.copy()

        # Indices of dimensions that should be transformed.
        idx = np.where(self.has_bounds)[0]

        if idx.size > 0:
            # 1) Normalize bounded components from [min_x, max_x] to [0, 1].
            x = (x[idx] - self.min_x[idx]) / (self.max_x[idx] - self.min_x[idx])

            # 2) Linearly map [0, 1] -> [-1, 1].
            x = (x * 2.0) - 1.0

            # 3) Clamp slightly away from -1 and 1 to avoid log / division
            #    singularities when applying the artanh-like mapping below.
            x = np.clip(x, -1.0 + self.eps, 1.0 - self.eps)

            # 4) Map [-1, 1] -> ℝ using the inverse tanh (artanh) formula:
            #       atanh(z) = 0.5 * log((1 + z) / (1 - z))
            #    This produces an unbounded representation for bounded inputs.
            y[idx] = 0.5 * np.log((1.0 + x) / (1.0 - x))

        # Unbounded or partially bounded dimensions are passed through as-is.
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
        """
        # Append the new bounds as length-1 arrays to maintain 1D structure.
        new_min_x = np.array([min_x], dtype=float)
        new_max_x = np.array([max_x], dtype=float)

        # Extend bounds vectors.
        self.min_x = np.concatenate([self.min_x, new_min_x])
        self.max_x = np.concatenate([self.max_x, new_max_x])

        # Recompute mask to keep internal invariants consistent.
        self.has_bounds = np.isfinite(self.min_x) & np.isfinite(self.max_x)
        return

    def remove_dimension(self, idx: int) -> None:
        """
        Remove one dimension and its bounds.

        Args:
            idx (int): Index of the dimension to remove.
        """
        # Remove the index across all internal arrays so they stay aligned.
        self.min_x = np.delete(self.min_x, idx, axis=0)
        self.max_x = np.delete(self.max_x, idx, axis=0)
        self.has_bounds = np.delete(self.has_bounds, idx, axis=0)
        return

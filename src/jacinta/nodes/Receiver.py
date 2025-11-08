from __future__ import annotations

import json

import numpy as np


class Receiver:
    """
    Receiver is responsible for mapping external inputs into an internal
    representation, optionally "unbounding" components that live inside
    finite intervals.

    It assumes that some dimensions may be:
        - fully bounded: [min_x[i], max_x[i]]
        - partially bounded: only one bound is finite
        - unbounded: no finite bounds

    Only fully bounded dimensions are transformed into ℝ via a
    logit-like mapping (based on artanh).
    """

    def __init__(
        self,
        size: int,
        min_x: float | np.ndarray | None = None,
        max_x: float | np.ndarray | None = None,
        eps: float = 1e-12,
    ) -> None:
        """
        Initialize a Receiver that rescales bounded signals.

        Args:
            size (int): Dimensionality of the input.
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

        # Normalize lower bounds input into a float array of shape (size,):
        # - None  -> all dimensions unbounded below    -> fill with NaN
        # - float -> same bound for all dimensions     -> broadcast scalar
        # - array -> per-dimension lower bounds        -> copy to avoid aliasing
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

        # A dimension is considered "bounded" only if both bounds are finite.
        # These are the only dimensions that will be transformed.
        self.has_bounds: np.ndarray = np.isfinite(self.min_x) & np.isfinite(self.max_x)
        return

    @property
    def N(self) -> int:
        """
        Return the dimensionality of the Receiver.

        Returns:
            int: Number of dimensions.
        """
        # Dimensionality is implicitly defined by the bounds vectors.
        N = self.min_x.size
        return N

    def copy(self) -> Receiver:
        """
        Create a deep copy of the current Receiver.

        Returns:
            Receiver: A new Receiver with the same parameters.
        """
        # Delegate the actual copying logic to the constructor to keep
        # the behavior consistent with __init__ (including dtype casting).
        receiver = self.__class__(self.N, self.min_x, self.max_x, self.eps)
        return receiver

    def to_dict(self) -> dict[str, any]:
        """
        Serialize Receiver configuration to a dictionary.

        Returns:
            dict[str, any]: Serializable snapshot of the Receiver.
        """
        # Convert numpy arrays into JSON-friendly Python lists.
        # Runtime class name is stored for potential introspection /
        # debugging, but reconstruction only relies on core fields.
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
    def from_dict(cls, data: dict[str, any]) -> Receiver:
        """
        Build a Receiver instance from a dict.

        Args:
            data (dict[str, any]): Serialized Receiver config.

        Returns:
            Receiver: Reconstructed Receiver instance.
        """
        # Rehydrate arrays and core scalar fields, then reuse __init__
        # to ensure any future validation / invariants stay centralized.
        receiver = cls(
            size=int(data["size"]),
            min_x=np.array(data["min_x"], dtype=float),
            max_x=np.array(data["max_x"], dtype=float),
            eps=float(data["eps"]),
        )
        return receiver

    def save(self, file_path: str) -> None:
        """
        Save Receiver configuration as JSON to given file.

        Args:
            file_path (str): Target file path.
        """
        # Persist only configuration, not any runtime state.
        data = self.to_dict()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        return

    @classmethod
    def load(cls, file_path: str) -> Receiver:
        """
        Load a Receiver instance from a JSON file.

        Args:
            file_path (str): Path to the JSON file containing saved configuration.

        Returns:
            Receiver: Reconstructed Receiver instance.
        """
        # Simple JSON deserialization path that mirrors `save`.
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        receiver = cls.from_dict(data)
        return receiver

    def process_forward(self, x: np.ndarray) -> np.ndarray:
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
            x = np.clip(x, -1 + self.eps, 1 - self.eps)

            # 4) Map [-1, 1] -> ℝ using the inverse tanh (artanh) formula:
            #       atanh(z) = 0.5 * log((1 + z) / (1 - z))
            #    This produces an unbounded representation for bounded inputs.
            y[idx] = 0.5 * np.log((1 + x) / (1 - x))

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
        # New bounds are appended as length-1 arrays to preserve the
        # 1D structure of min_x / max_x.
        new_min_x = np.array([min_x], dtype=float)
        new_max_x = np.array([max_x], dtype=float)

        # Concatenate along the single dimension axis.
        self.min_x = np.concatenate([self.min_x, new_min_x])
        self.max_x = np.concatenate([self.max_x, new_max_x])

        # Recompute the mask to keep invariants consistent after resizing.
        self.has_bounds = np.isfinite(self.min_x) & np.isfinite(self.max_x)
        return

    def remove_dimension(self, idx: int) -> None:
        """
        Remove one dimension and its bounds.

        Args:
            idx (int): Index of the dimension to remove.
        """
        # Remove the selected index from all internal arrays to keep
        # them aligned dimension-wise.
        self.min_x = np.delete(self.min_x, idx, axis=0)
        self.max_x = np.delete(self.max_x, idx, axis=0)
        self.has_bounds = np.delete(self.has_bounds, idx, axis=0)
        return

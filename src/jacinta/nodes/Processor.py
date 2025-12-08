from __future__ import annotations

import json
import os
from typing import Any

import numpy as np


class Processor:
    """
    Stochastic Gaussian policy over a continuous action space.

    The policy is represented as a multivariate Gaussian N(mu, Sigma) and
    can:
        - sample actions (process_forward)
        - update its parameters from rewards (process_backward)
    """

    def __init__(
        self,
        size: int,
        mu: float | np.ndarray = 0.0,
        sigma: float | np.ndarray = 10.0,
        lr_mu: float = 1e-3,
        lr_sigma: float = 1e-4,
        min_var: float = 1e-8,
        r_alpha: float = 1e-2,
        r_beta: float = 1e-2,
        eps: float = 1e-12,
    ) -> None:
        """
        Initialize a Processor with a Gaussian policy N(mu, Sigma).

        Args:
            size (int): Dimensionality of the process (number of dimensions).
            mu (float | np.ndarray): Initial mean:
                - scalar: broadcast to all dimensions
                - 1D array: per-dimension mean, length must be `size`
            sigma (float | np.ndarray): Initial spread:
                - scalar: global variance, Sigma = sigma * I
                - 1D array: per-dimension variance, diagonal covariance
                - 2D array: full covariance matrix (size x size)
            lr_mu (float): Learning rate for mean updates.
            lr_sigma (float): Learning rate for covariance updates.
            min_var (float): Minimum allowed variance per dimension.
            r_alpha (float): EMA factor for reward baseline updates.
            r_beta (float): EMA factor for reward scale updates.
            eps (float): Numerical epsilon for stability (division, norms).
        """
        # Validate and normalize dimensionality.
        if not isinstance(size, (int, np.integer)):
            raise TypeError("size must be an integer")
        if size <= 0:
            raise ValueError("size must be positive")
        size = int(size)

        # Validate and initialize mean.
        if not isinstance(
            mu, (int, float, np.integer, np.floating, np.ndarray, list, tuple)
        ):
            raise TypeError(
                "mu must be a numeric scalar or a 1D array-like of numerics"
            )

        if np.isscalar(mu):
            # Scalar mean, broadcast to all dimensions.
            self.mu = np.full(size, float(mu), dtype=float)
        else:
            # 1D array mean, must match size.
            mu = np.asarray(mu)
            if mu.ndim != 1:
                raise ValueError("mu must be a 1D array")
            if mu.size != size:
                raise ValueError(f"mu must have size {size}")
            if not np.issubdtype(mu.dtype, np.number):
                raise TypeError("mu must be numeric")
            self.mu = mu.astype(float, copy=True)

        # Validate and initialize covariance.
        if not isinstance(
            sigma, (int, float, np.integer, np.floating, np.ndarray, list, tuple)
        ):
            raise TypeError("sigma must be a numeric scalar or a numeric array-like")

        if np.isscalar(sigma):
            # Scalar sigma interpreted as global variance, Sigma = sigma * I.
            sigma = float(sigma)
            if sigma < 0.0:
                raise ValueError("scalar sigma (variance) must be non-negative")
            self.sigma = sigma * np.eye(size, dtype=float)
        else:
            # Array sigma can be 1D (variances) or 2D (full covariance).
            sigma = np.asarray(sigma)
            if not (1 <= sigma.ndim <= 2):
                raise ValueError("sigma must be a 1D or 2D array")
            if not np.issubdtype(sigma.dtype, np.number):
                raise TypeError("sigma must be numeric")
            if sigma.ndim == 1:
                # Per-dimension variances on the diagonal, no correlations.
                if sigma.size != size:
                    raise ValueError(f"sigma must have size {size}")
                sigma_diag = sigma.astype(float, copy=True)
                if not np.all(sigma_diag >= 0.0):
                    raise ValueError("1D sigma entries (variances) must be >= 0")
                self.sigma = np.diag(sigma_diag)
            else:
                # Full covariance matrix, symmetrized for numerical stability.
                if sigma.shape != (size, size):
                    raise ValueError(f"sigma must have shape ({size}, {size})")
                self.sigma = 0.5 * (sigma + sigma.T).astype(float, copy=False)

        # Validate and store minimum variance.
        if not isinstance(min_var, (int, float, np.integer, np.floating)):
            raise TypeError("min_var must be a float")
        if min_var < 0:
            raise ValueError("min_var must be non-negative")
        self.min_var = float(min_var)

        # Enforce minimum variance on the covariance diagonal.
        sigma_diag = np.diag(self.sigma)
        sigma_diag = np.maximum(sigma_diag, self.min_var)
        np.fill_diagonal(self.sigma, sigma_diag)

        # Ensure covariance is positive definite (or project it if needed).
        self.sigma = self._project_to_spd(self.sigma)

        # Validate and store epsilon.
        if not isinstance(eps, (int, float, np.integer, np.floating)):
            raise TypeError("eps must be a float")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = float(eps)

        # Validate and store learning rates.
        if not isinstance(lr_mu, (int, float, np.integer, np.floating)):
            raise TypeError("lr_mu must be a float")
        if lr_mu < 0:
            raise ValueError("lr_mu must be non-negative")
        self.lr_mu = float(lr_mu)

        if not isinstance(lr_sigma, (int, float, np.integer, np.floating)):
            raise TypeError("lr_sigma must be a float")
        if lr_sigma < 0:
            raise ValueError("lr_sigma must be non-negative")
        self.lr_sigma = float(lr_sigma)

        # Validate and store reward EMA factors.
        if not isinstance(r_alpha, (int, float, np.integer, np.floating)):
            raise TypeError("r_alpha must be a float")
        if not (0 <= r_alpha <= 1.0):
            raise ValueError("r_alpha must be in [0, 1]")
        self.r_alpha = float(r_alpha)

        if not isinstance(r_beta, (int, float, np.integer, np.floating)):
            raise TypeError("r_beta must be a float")
        if not (0 <= r_beta <= 1.0):
            raise ValueError("r_beta must be in [0, 1]")
        self.r_beta = float(r_beta)

        # Reward normalization state (scalar EMA statistics).
        self.r_baseline: float = 0.0
        self.r_scale: float = 1.0
        return

    @property
    def N(self) -> int:
        """
        Dimensionality of the process.

        Returns:
            int: Number of dimensions of the Gaussian policy.
        """
        # Dimensionality is the length of the mean vector.
        N = self.mu.size
        return N

    def copy(self) -> Processor:
        """
        Create a deep copy of the Processor.

        Returns:
            Processor: New instance with the same parameters and state.
        """
        # Reuse the constructor to preserve all invariants.
        processor = self.__class__(
            self.N,
            self.mu,
            self.sigma,
            self.lr_mu,
            self.lr_sigma,
            self.min_var,
            self.r_alpha,
            self.r_beta,
            self.eps,
        )

        # Copy reward normalization scalars.
        processor.r_baseline = self.r_baseline
        processor.r_scale = self.r_scale
        return processor

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize Processor state to a Python dictionary.

        Returns:
            dict[str, Any]: Serializable snapshot of the Processor.
        """
        # Convert numpy arrays to lists so they can be JSON-encoded.
        data = {
            "class": self.__class__.__name__,
            "size": self.N,
            "mu": self.mu.tolist(),
            "sigma": self.sigma.tolist(),
            "lr_mu": self.lr_mu,
            "lr_sigma": self.lr_sigma,
            "min_var": self.min_var,
            "r_alpha": self.r_alpha,
            "r_beta": self.r_beta,
            "eps": self.eps,
            "r_baseline": self.r_baseline,
            "r_scale": self.r_scale,
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Processor:
        """
        Reconstruct a Processor from a serialized dictionary.

        Args:
            data (dict[str, Any]): Serialized Processor configuration.

        Returns:
            Processor: Reconstructed Processor instance.
        """
        # Validate the outer container and required keys.
        if not isinstance(data, dict):
            raise TypeError("data must be a dictionary")
        required_keys = {
            "class",
            "size",
            "mu",
            "sigma",
            "lr_mu",
            "lr_sigma",
            "min_var",
            "r_alpha",
            "r_beta",
            "eps",
            "r_baseline",
            "r_scale",
        }
        if set(data.keys()) != required_keys:
            raise ValueError(
                "data must have exactly the keys: "
                "class, size, mu, sigma, lr_mu, lr_sigma, "
                "min_var, r_alpha, r_beta, eps, r_baseline, r_scale"
            )

        # Ensure class name matches the current class.
        if not isinstance(data["class"], str):
            raise TypeError("data['class'] must be a string")
        if data["class"] != cls.__name__:
            raise ValueError(f"data['class'] must be {cls.__name__}")

        # Delegate parameter validation to __init__.
        processor = cls(
            size=int(data["size"]),
            mu=np.array(data["mu"], dtype=float),
            sigma=np.array(data["sigma"], dtype=float),
            lr_mu=float(data["lr_mu"]),
            lr_sigma=float(data["lr_sigma"]),
            min_var=float(data["min_var"]),
            r_alpha=float(data["r_alpha"]),
            r_beta=float(data["r_beta"]),
            eps=float(data["eps"]),
        )

        # Restore reward normalization scalars.
        if not isinstance(data["r_baseline"], (int, float, np.integer, np.floating)):
            raise TypeError("r_baseline must be numeric")
        processor.r_baseline = float(data["r_baseline"])

        if not isinstance(data["r_scale"], (int, float, np.integer, np.floating)):
            raise TypeError("r_scale must be numeric")
        if data["r_scale"] < 0:
            raise ValueError("r_scale must be non-negative")
        processor.r_scale = float(data["r_scale"])
        return processor

    def save(self, file_path: str) -> None:
        """
        Save Processor state as JSON to a file.

        Args:
            file_path (str): Target JSON file path.
        """
        # Validate file path type.
        if not isinstance(file_path, str):
            raise TypeError("file_path must be a string")

        # Validate directory if present (empty means current directory).
        file_dir = os.path.dirname(file_path)
        if file_dir and not os.path.isdir(file_dir):
            raise FileNotFoundError(f"directory {file_dir} does not exist")

        # Serialize and write JSON to disk.
        data = self.to_dict()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return

    @classmethod
    def load(cls, file_path: str) -> Processor:
        """
        Load a Processor from a JSON file.

        Args:
            file_path (str): Path to the JSON file containing saved state.

        Returns:
            Processor: Reconstructed Processor instance.
        """
        # Validate file path and existence.
        if not isinstance(file_path, str):
            raise TypeError("file_path must be a string")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"file {file_path} does not exist")

        # Read JSON and delegate to from_dict.
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        processor = cls.from_dict(data)
        return processor

    def process_forward(self, n: int = 1) -> np.ndarray:
        """
        Sample actions from the current Gaussian policy.

        Args:
            n (int): Number of samples to draw (n >= 0).

        Returns:
            np.ndarray: Array of shape (n, N) with samples from N(mu, sigma).
        """
        # Validate and normalize number of samples.
        if not isinstance(n, (int, np.integer)):
            raise TypeError("n must be an integer")
        if n < 0:
            raise ValueError("n must be non-negative")
        n = int(n)

        # Cholesky factorization: sigma = L L^T, requires sigma to be SPD.
        L = np.linalg.cholesky(self.sigma)

        # Draw standard normal noise in R^N.
        z = np.random.randn(n, self.N)

        # Reparameterization: y = mu + z L^T, where z ~ N(0, I).
        y = self.mu + z @ L.T
        return y

    def process_backward(self, y: np.ndarray, r: np.ndarray) -> None:
        """
        Batch update of mean and covariance using samples and rewards.

        Args:
            y (np.ndarray): Batch of samples, shape (batch_size, N).
            r (np.ndarray): Batch of rewards, shape (batch_size,).
        """
        # Validate array types.
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a numpy array")
        if not isinstance(r, np.ndarray):
            raise TypeError("r must be a numpy array")

        # Validate shapes: y is 2D, r is 1D.
        if y.ndim != 2:
            raise ValueError("y must be a 2D array")
        if r.ndim != 1:
            raise ValueError("r must be a 1D array")

        # Validate dimensions: second axis of y must match N.
        if y.shape[1] != self.N:
            raise ValueError(f"y must have shape (n, {self.N})")
        # Validate batch size consistency.
        if r.size != y.shape[0]:
            raise ValueError("r and y must have the same batch size")

        # Ensure numeric, float arrays for computations.
        if not np.issubdtype(y.dtype, np.number):
            raise TypeError("y must be a numeric array")
        y = y.astype(float, copy=True)

        if not np.issubdtype(r.dtype, np.number):
            raise TypeError("r must be a numeric array")
        r = r.astype(float, copy=True)

        # Apply single-sample update for each (y_i, r_i) pair.
        for yi, ri in zip(y, r, strict=True):
            self._process_backward(yi, ri)
        return

    def _process_backward(self, y: np.ndarray, r: float) -> None:
        """
        Single-sample update of mean and covariance.

        Args:
            y (np.ndarray): Sample vector of shape (N,).
            r (float): Reward associated with this sample.
        """
        # Normalize reward before applying the update.
        r = self._process_reward(r)

        # Deviation from current mean.
        diff = y - self.mu

        # Mean update direction: solve sigma * v_mu = diff (sigma^{-1} diff).
        try:
            v_mu = np.linalg.solve(self.sigma, diff)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if sigma is ill-conditioned.
            v_mu = np.linalg.pinv(self.sigma) @ diff

        # Normalize direction to decouple magnitude from diff norm.
        v_mu /= np.linalg.norm(v_mu) + self.eps

        # Apply mean update scaled by normalized reward.
        self.mu += self.lr_mu * r * v_mu

        # Covariance update direction: push towards outer(diff, diff).
        v_sigma = np.outer(diff, diff) - self.sigma

        # Normalize covariance update to avoid large steps.
        v_sigma /= np.linalg.norm(v_sigma) + self.eps

        # Apply covariance update.
        self.sigma += self.lr_sigma * r * v_sigma

        # Enforce exact symmetry of sigma.
        self.sigma = 0.5 * (self.sigma + self.sigma.T)

        # Enforce minimum variance on the diagonal to preserve SPD.
        sigma_diag = np.diag(self.sigma)
        sigma_diag = np.maximum(sigma_diag, self.min_var)
        np.fill_diagonal(self.sigma, sigma_diag)

        # Project updated covariance back to SPD if needed.
        self.sigma = self._project_to_spd(self.sigma)
        return

    def add_dimension(
        self, mu: float | np.ndarray = 0.0, sigma: float | np.ndarray = 100.0
    ) -> None:
        """
        Add one or more dimensions to the Gaussian policy.

        Args:
            mu (float | np.ndarray): Mean for new dimensions:
                - scalar: adds one dimension with that mean
                - 1D array: per-dimension mean for new dimensions
            sigma (float | np.ndarray): Spread for new dimensions:
                - scalar: global variance, diagonal covariance
                - 1D array: per-dimension variance, diagonal covariance
                - 2D array: full covariance matrix for new block
        """
        # Validate and normalize new mean components.
        if not isinstance(
            mu, (int, float, np.integer, np.floating, np.ndarray, list, tuple)
        ):
            raise TypeError(
                "mu must be a numeric scalar or a 1D array-like of numerics"
            )

        if np.isscalar(mu):
            # Single new dimension with scalar mean.
            mu = np.full(1, float(mu), dtype=float)
        else:
            # Multiple new dimensions from a 1D array.
            mu = np.asarray(mu)
            if mu.ndim != 1:
                raise ValueError("mu must be a 1D array")
            if mu.size <= 0:
                raise ValueError("mu must have at least one element")
            if not np.issubdtype(mu.dtype, np.number):
                raise TypeError("mu must be numeric")
            mu = mu.astype(float, copy=True)

        # Validate and normalize new covariance block.
        if not isinstance(
            sigma, (int, float, np.integer, np.floating, np.ndarray, list, tuple)
        ):
            raise TypeError("sigma must be a numeric scalar or a numeric array-like")

        if np.isscalar(sigma):
            # Scalar sigma interpreted as global variance for new dimensions.
            sigma = float(sigma)
            if sigma < 0.0:
                raise ValueError("scalar sigma (variance) must be non-negative")
            sigma = sigma * np.eye(mu.size, dtype=float)
        else:
            sigma = np.asarray(sigma)
            if not (1 <= sigma.ndim <= 2):
                raise ValueError("sigma must be a 1D or 2D array")
            if not np.issubdtype(sigma.dtype, np.number):
                raise TypeError("sigma must be numeric")
            if sigma.ndim == 1:
                # 1D sigma interpreted as variances on the diagonal.
                if sigma.size != mu.size:
                    raise ValueError(f"sigma must have size {mu.size}")
                sigma_diag = sigma.astype(float, copy=True)
                if not np.all(sigma_diag >= 0.0):
                    raise ValueError("1D sigma entries (variances) must be >= 0")
                sigma = np.diag(sigma_diag)
            else:
                # 2D full covariance for the new block.
                if sigma.shape != (mu.size, mu.size):
                    raise ValueError(f"sigma must have shape ({mu.size}, {mu.size})")
                sigma = 0.5 * (sigma + sigma.T).astype(float, copy=False)

        # Enforce minimum variance on the new block diagonal.
        sigma_diag = np.diag(sigma)
        sigma_diag = np.maximum(sigma_diag, self.min_var)
        np.fill_diagonal(sigma, sigma_diag)

        # Ensure the new covariance block is positive definite (project if needed).
        sigma = self._project_to_spd(sigma)

        # Create enlarged covariance matrix with block-diagonal structure.
        new_sigma = np.zeros((self.N + mu.size, self.N + mu.size), dtype=float)
        new_sigma[: self.N, : self.N] = self.sigma
        new_sigma[self.N :, self.N :] = sigma

        # Concatenate means and update covariance.
        self.mu = np.concatenate([self.mu, mu])
        self.sigma = new_sigma
        return

    def remove_dimension(self, idx: int | np.ndarray) -> None:
        """
        Remove one or more dimensions from the Gaussian policy.

        Args:
            idx (int | np.ndarray): Index or 1D array of indices to remove.
                Negative indices are supported (Python-style).
        """
        # Validate index container type.
        if not isinstance(idx, (int, np.integer, np.ndarray, list, tuple)):
            raise TypeError("idx must be an integer or a 1D array-like of integers")

        if np.isscalar(idx):
            # Single index case.
            idx = np.array([idx], dtype=int)
        else:
            # Multiple indices case.
            idx = np.asarray(idx)
            if idx.ndim != 1:
                raise ValueError("idx must be a 1D array")
            if not np.issubdtype(idx.dtype, np.integer):
                raise TypeError("idx must contain integer indices")

        # Disallow empty index sets and removing all dimensions.
        if idx.size <= 0:
            raise ValueError("idx must have at least one element")
        if idx.size >= self.N:
            raise ValueError(
                f"idx must have less than {self.N} elements (cannot remove all dimensions)"
            )

        # Normalize negative indices relative to current dimensionality.
        idx[idx < 0] += self.N

        # Ensure indices are within valid range.
        if not np.all(idx >= 0):
            raise ValueError("all indices in idx must be >= 0")
        if not np.all(idx < self.N):
            raise ValueError(f"all indices in idx must be < {self.N}")

        # Remove duplicate indices and sort them.
        idx = np.unique(idx)

        # Remove selected dimensions from mean and covariance.
        self.mu = np.delete(self.mu, idx, axis=0)
        self.sigma = np.delete(self.sigma, idx, axis=0)
        self.sigma = np.delete(self.sigma, idx, axis=1)
        return

    def _process_reward(self, r: float) -> float:
        """
        Normalize a raw reward using running baseline and scale.

        Args:
            r (float): Raw reward value.

        Returns:
            float: Normalized reward with approximately unit scale.
        """
        r = float(r)

        # EMA update for reward baseline.
        self.r_baseline = (1.0 - self.r_alpha) * self.r_baseline + self.r_alpha * r

        # Center reward around the baseline.
        r_centered = r - self.r_baseline

        # EMA update for reward scale (approximate magnitude).
        self.r_scale = (1.0 - self.r_beta) * self.r_scale + self.r_beta * abs(
            r_centered
        )

        # Normalize reward using scale and epsilon for stability.
        r_norm = r_centered / (self.r_scale + self.eps)
        return r_norm

    def _project_to_spd(self, sigma: np.ndarray) -> np.ndarray:
        """
        Project a symmetric matrix onto the space of SPD matrices.

        Args:
            sigma (np.ndarray): Symmetric covariance candidate.

        Returns:
            np.ndarray: Symmetric positive definite covariance matrix.
        """
        # Symmetrize explicitly to avoid asymmetries from numerical noise.
        sigma = 0.5 * (sigma + sigma.T)

        # Fast path: try Cholesky first.
        try:
            np.linalg.cholesky(sigma)
        except np.linalg.LinAlgError:
            # Eigen-decomposition and eigenvalue clipping.
            eigvals, eigvecs = np.linalg.eigh(sigma)
            # Clamp eigenvalues to at least min_var for positive definiteness.
            eigvals_clamped = np.maximum(eigvals, self.min_var)
            sigma = eigvecs @ np.diag(eigvals_clamped) @ eigvecs.T

            # Re-symmetrize to clean numerical noise.
            sigma = 0.5 * (sigma + sigma.T)

            # Final check: if this still fails, raise a hard error.
            try:
                np.linalg.cholesky(sigma)
            except np.linalg.LinAlgError as exc:
                raise RuntimeError(
                    "Failed to project sigma to a positive definite covariance"
                ) from exc

        return sigma

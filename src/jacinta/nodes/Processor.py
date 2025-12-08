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
        assert isinstance(size, (int, np.integer)), "size must be an integer"
        assert size > 0, "size must be positive"
        size = int(size)

        # Validate and initialize mean.
        assert isinstance(
            mu, (int, float, np.integer, np.floating, np.ndarray, list, tuple)
        ), "mu must be a float or 1D array of floats"

        if np.isscalar(mu):
            # Scalar mean, broadcast to all dimensions.
            self.mu = np.full(size, float(mu), dtype=float)
        else:
            # 1D array mean, must match size.
            mu = np.asarray(mu)
            assert mu.ndim == 1, "mu must be a 1D array"
            assert mu.size == size, f"mu must have size {size}"
            assert np.issubdtype(mu.dtype, np.number), "mu must be numeric"
            self.mu = mu.astype(float, copy=True)

        # Validate and initialize covariance.
        assert isinstance(
            sigma, (int, float, np.integer, np.floating, np.ndarray, list, tuple)
        ), "sigma must be a float or array of floats"

        if np.isscalar(sigma):
            # Scalar sigma interpreted as global variance, Sigma = sigma * I.
            sigma = float(sigma)
            assert sigma >= 0.0, "scalar sigma (variance) must be non-negative"
            self.sigma = sigma * np.eye(size, dtype=float)
        else:
            # Array sigma can be 1D (variances) or 2D (full covariance).
            sigma = np.asarray(sigma)
            assert 1 <= sigma.ndim <= 2, "sigma must be a 1D or 2D array"
            assert np.issubdtype(sigma.dtype, np.number), "sigma must be numeric"
            if sigma.ndim == 1:
                # Per-dimension variances on the diagonal, no correlations.
                assert sigma.size == size, f"sigma must have size {size}"
                sigma_diag = sigma.astype(float, copy=True)
                assert np.all(
                    sigma_diag >= 0.0
                ), "1D sigma entries (variances) must be >= 0"
                self.sigma = np.diag(sigma_diag)
            else:
                # Full covariance matrix, symmetrized for numerical stability.
                assert sigma.shape == (
                    size,
                    size,
                ), f"sigma must have shape ({size}, {size})"
                self.sigma = 0.5 * (sigma + sigma.T).astype(float, copy=False)

        # Validate and store minimum variance.
        assert isinstance(
            min_var, (int, float, np.integer, np.floating)
        ), "min_var must be a float"
        assert min_var >= 0, "min_var must be non-negative"
        self.min_var = float(min_var)

        # Enforce minimum variance on the covariance diagonal.
        sigma_diag = np.diag(self.sigma)
        sigma_diag = np.maximum(sigma_diag, self.min_var)
        np.fill_diagonal(self.sigma, sigma_diag)

        # Ensure covariance is positive definite (required by Cholesky).
        try:
            np.linalg.cholesky(self.sigma)
        except np.linalg.LinAlgError as exc:
            raise AssertionError(
                "sigma must define a positive definite covariance"
            ) from exc

        # Validate and store epsilon.
        assert isinstance(
            eps, (int, float, np.integer, np.floating)
        ), "eps must be a float"
        assert eps > 0, "eps must be positive"
        self.eps = float(eps)

        # Validate and store learning rates.
        assert isinstance(
            lr_mu, (int, float, np.integer, np.floating)
        ), "lr_mu must be a float"
        assert lr_mu >= 0, "lr_mu must be non-negative"
        self.lr_mu = float(lr_mu)

        assert isinstance(
            lr_sigma, (int, float, np.integer, np.floating)
        ), "lr_sigma must be a float"
        assert lr_sigma >= 0, "lr_sigma must be non-negative"
        self.lr_sigma = float(lr_sigma)

        # Validate and store reward EMA factors.
        assert isinstance(
            r_alpha, (int, float, np.integer, np.floating)
        ), "r_alpha must be a float"
        assert 0 <= r_alpha <= 1.0, "r_alpha must be in [0, 1]"
        self.r_alpha = float(r_alpha)

        assert isinstance(
            r_beta, (int, float, np.integer, np.floating)
        ), "r_beta must be a float"
        assert 0 <= r_beta <= 1.0, "r_beta must be in [0, 1]"
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
        assert isinstance(data, dict), "data must be a dictionary"
        assert data.keys() == {
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
        }, "data must have keys: class, size, mu, sigma, lr_mu, lr_sigma, min_var, r_alpha, r_beta, eps, r_baseline, r_scale"

        # Ensure class name matches the current class.
        assert isinstance(data["class"], str), "class must be a string"
        assert data["class"] == cls.__name__, f"class must be {cls.__name__}"

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
        assert isinstance(
            data["r_baseline"], (int, float, np.integer, np.floating)
        ), "r_baseline must be a float"
        processor.r_baseline = float(data["r_baseline"])

        assert isinstance(
            data["r_scale"], (int, float, np.integer, np.floating)
        ), "r_scale must be a float"
        assert data["r_scale"] >= 0, "r_scale must be non-negative"
        processor.r_scale = float(data["r_scale"])
        return processor

    def save(self, file_path: str) -> None:
        """
        Save Processor state as JSON to a file.

        Args:
            file_path (str): Target JSON file path.
        """
        # Validate file path type.
        assert isinstance(file_path, str), "file_path must be a string"

        # Validate directory if present (empty means current directory).
        file_dir = os.path.dirname(file_path)
        if file_dir:
            assert os.path.exists(file_dir), f"directory {file_dir} does not exist"

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
        assert isinstance(file_path, str), "file_path must be a string"
        assert os.path.isfile(file_path), f"file {file_path} does not exist"

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
        assert isinstance(n, (int, np.integer)), "n must be an integer"
        assert n >= 0, "n must be non-negative"
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
        assert isinstance(y, np.ndarray), "y must be a numpy array"
        assert isinstance(r, np.ndarray), "r must be a numpy array"

        # Validate shapes: y is 2D, r is 1D.
        assert y.ndim == 2, "y must be a 2D array"
        assert r.ndim == 1, "r must be a 1D array"

        # Validate dimensions: second axis of y must match N.
        assert y.shape[1] == self.N, f"y must have shape (n, {self.N})"
        # Validate batch size consistency.
        assert r.size == y.shape[0], "r and y must have the same batch size"

        # Ensure numeric, float arrays for computations.
        assert np.issubdtype(y.dtype, np.number), "y must be a floating array"
        y = y.astype(float, copy=True)

        assert np.issubdtype(r.dtype, np.number), "r must be a floating array"
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
        assert isinstance(
            mu, (int, float, np.integer, np.floating, np.ndarray, list, tuple)
        ), "mu must be a float or 1D array of floats"

        if np.isscalar(mu):
            # Single new dimension with scalar mean.
            mu = np.full(1, float(mu), dtype=float)
        else:
            # Multiple new dimensions from a 1D array.
            mu = np.asarray(mu)
            assert mu.ndim == 1, "mu must be a 1D array"
            assert mu.size > 0, "mu must have at least one element"
            assert np.issubdtype(mu.dtype, np.number), "mu must be numeric"
            mu = mu.astype(float, copy=True)

        # Validate and normalize new covariance block.
        assert isinstance(
            sigma, (int, float, np.integer, np.floating, np.ndarray, list, tuple)
        ), "sigma must be a float or array of floats"

        if np.isscalar(sigma):
            # Scalar sigma interpreted as global variance for new dimensions.
            sigma = float(sigma)
            assert sigma >= 0.0, "scalar sigma (variance) must be non-negative"
            sigma = sigma * np.eye(mu.size, dtype=float)
        else:
            sigma = np.asarray(sigma)
            assert 1 <= sigma.ndim <= 2, "sigma must be a 1D or 2D array"
            assert np.issubdtype(sigma.dtype, np.number), "sigma must be numeric"
            if sigma.ndim == 1:
                # 1D sigma interpreted as variances on the diagonal.
                assert sigma.size == mu.size, f"sigma must have size {mu.size}"
                sigma_diag = sigma.astype(float, copy=True)
                assert np.all(
                    sigma_diag >= 0.0
                ), "1D sigma entries (variances) must be >= 0"
                sigma = np.diag(sigma_diag)
            else:
                # 2D full covariance for the new block.
                assert sigma.shape == (
                    mu.size,
                    mu.size,
                ), f"sigma must have shape ({mu.size}, {mu.size})"
                sigma = 0.5 * (sigma + sigma.T).astype(float, copy=False)

        # Enforce minimum variance on the new block diagonal.
        sigma_diag = np.diag(sigma)
        sigma_diag = np.maximum(sigma_diag, self.min_var)
        np.fill_diagonal(sigma, sigma_diag)

        # Ensure the new covariance block is positive definite.
        try:
            np.linalg.cholesky(sigma)
        except np.linalg.LinAlgError as exc:
            raise AssertionError(
                "sigma must define a positive definite covariance"
            ) from exc

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
        assert isinstance(
            idx, (int, np.integer, np.ndarray, list, tuple)
        ), "idx must be an integer or array of integers"

        if np.isscalar(idx):
            # Single index case.
            idx = np.array([idx], dtype=int)
        else:
            # Multiple indices case.
            idx = np.asarray(idx)
            assert idx.ndim == 1, "idx must be a 1D array"
            assert np.issubdtype(idx.dtype, np.integer), "idx must be an integer array"

        # Disallow empty index sets and removing all dimensions.
        assert idx.size > 0, "idx must have at least one element"
        assert idx.size < self.N, f"idx must have less than {self.N} elements"

        # Normalize negative indices relative to current dimensionality.
        idx[idx < 0] += self.N

        # Ensure indices are within valid range.
        assert np.all(idx >= 0), "idx must be non-negative"
        assert np.all(idx < self.N), f"idx must be less than {self.N}"

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

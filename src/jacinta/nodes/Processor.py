from __future__ import annotations

import json
from typing import Any

import numpy as np


class Processor:
    """ """

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
        """ """

        assert isinstance(size, (int, np.integer)), "size must be an integer"
        assert size > 0, "size must be positive"
        size = int(size)

        assert isinstance(
            mu, (int, float, np.integer, np.floating, np.ndarray, list, tuple)
        ), "mu must be a float or 1D array of floats"

        if np.isscalar(mu):
            self.mu = np.full(size, float(mu), dtype=float)
        else:
            mu = np.asarray(mu)
            assert mu.ndim == 1, "mu must be a 1D array"
            assert mu.size == size, f"mu must have size {size}"
            assert np.issubdtype(mu.dtype, np.number), "mu must be numeric"
            self.mu = mu.astype(float, copy=True)

        assert isinstance(
            sigma, (int, float, np.integer, np.floating, np.ndarray, list, tuple)
        ), "sigma must be a float or array of floats"

        if np.isscalar(sigma):
            sigma = float(sigma)
            assert sigma >= 0.0, "scalar sigma (std) must be non-negative"
            self.sigma = (sigma**2) * np.eye(size, dtype=float)

        else:
            sigma = np.asarray(sigma)
            assert 1 <= sigma.ndim <= 2, "sigma must be a 1D or 2D array"
            assert np.issubdtype(sigma.dtype, np.number), "sigma must be numeric"
            if sigma.ndim == 1:
                assert sigma.size == size, f"sigma must have size {size}"
                sigma_diag = sigma.astype(float, copy=True)
                assert np.all(
                    sigma_diag >= 0.0
                ), "1D sigma entries (variances) must be >= 0"
                self.sigma = np.diag(sigma_diag)
            else:
                assert sigma.shape == (
                    size,
                    size,
                ), f"sigma must have shape ({size}, {size})"
                self.sigma = 0.5 * (sigma + sigma.T).astype(float, copy=False)

        assert isinstance(
            min_var, (int, float, np.integer, np.floating)
        ), "min_var must be a float"
        assert min_var >= 0, "min_var must be non-negative"
        self.min_var = float(min_var)

        sigma_diag = np.diag(self.sigma)
        sigma_diag = np.maximum(sigma_diag, self.min_var)
        np.fill_diagonal(self.sigma, sigma_diag)

        try:
            np.linalg.cholesky(self.sigma)
        except np.linalg.LinAlgError as exc:
            raise AssertionError(
                "sigma must define a positive definite covariance"
            ) from exc

        assert isinstance(
            eps, (int, float, np.integer, np.floating)
        ), "eps must be a float"
        assert eps > 0, "eps must be positive"
        self.eps = float(eps)

        assert isinstance(
            lr_mu, (int, float, np.integer, np.floating)
        ), "lr_mu must be a float"
        assert lr_mu > 0, "lr_mu must be positive"
        self.lr_mu = float(lr_mu)

        assert isinstance(
            lr_sigma, (int, float, np.integer, np.floating)
        ), "lr_sigma must be a float"
        assert lr_sigma > 0, "lr_sigma must be positive"
        self.lr_sigma = float(lr_sigma)

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

        self.r_baseline: float = 0.0
        self.r_scale: float = 1.0
        return

    @property
    def N(self) -> int:
        """
        Return the dimensionality of the Processor.

        Returns:
            int: Number of dimensions.
        """
        # Dimensionality is inferred from the length of the mean vector.
        N = self.mu.size
        return N

    def copy(self) -> Processor:
        """
        Create a deep copy of the current Processor.

        Returns:
            Processor: A new Processor with the same parameters.
        """
        # Reuse constructor to keep behavior consistent with __init__.
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

        # Copy reward normalization state.
        processor.r_baseline = self.r_baseline
        processor.r_scale = self.r_scale
        return processor

    ################################################################
    ## Pending Refactor
    ################################################################

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize Processor state to a dictionary.

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
            "r_baseline": self.r_baseline.tolist(),
            "r_scale": self.r_scale.tolist(),
            "r_alpha": self.r_alpha,
            "r_beta": self.r_beta,
            "eps": self.eps,
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Processor:
        """
        Build a Processor instance from a dict.

        Args:
            data (dict[str, Any]): Serialized Processor config.

        Returns:
            Processor: Reconstructed Processor instance.
        """
        # Rehydrate numpy arrays and delegate to __init__ for validation.
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

        # Restore reward normalization state.
        processor.r_baseline = np.array(data["r_baseline"], dtype=float)
        processor.r_scale = np.array(data["r_scale"], dtype=float)
        return processor

    def save(self, file_path: str) -> None:
        """
        Save Processor state as JSON to given file.

        Args:
            file_path (str): Target file path.
        """
        # Persist configuration and learning state for later reuse.
        data = self.to_dict()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return

    @classmethod
    def load(cls, file_path: str) -> Processor:
        """
        Load a Processor instance from a JSON file.

        Args:
            file_path (str): Path to the JSON file containing saved state.

        Returns:
            Processor: Reconstructed Processor instance.
        """
        # Simple deserialization path mirroring save().
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        processor = cls.from_dict(data)
        return processor

    def process_forward(self, n: int = 1) -> np.ndarray:
        """
        Sample from the current Gaussian distribution.

        Args:
            n (int): Number of samples.

        Returns:
            np.ndarray: Array of shape (n, N) with samples from N(mu, sigma).
        """
        # Cholesky factorization: sigma = L L.T, assumes sigma is SPD.
        L = np.linalg.cholesky(self.sigma)

        # Standard normal noise in R^N.
        z = np.random.randn(n, self.N)

        # Reparameterization: y = mu + z L.T : z ~ N(0, I).
        y = self.mu + z @ L.T
        return y

    def process_backward(self, y: np.ndarray, r: np.ndarray) -> None:
        """
        Update mean and covariance based on a sample and reward signal.

        Args:
            y (np.ndarray): Sampled point used to guide the update.
            r (np.ndarray): Reward signal used to scale the update.
        """
        # Normalize reward to stabilize learning across different scales.
        r = self._process_reward(r)

        # Deviation of sample from current mean.
        diff = y - self.mu

        # Direction for mean update:
        # solve sigma * v_mu = diff  -> v_mu = sigma^{-1} diff
        # fall back to pseudo-inverse if sigma is near-singular.
        try:
            v_mu = np.linalg.solve(self.sigma, diff)
        except np.linalg.LinAlgError:
            v_mu = np.linalg.pinv(self.sigma) @ diff

        # Normalize direction to unit length to decouple update magnitude
        # from the norm of diff / sigma.
        v_mu /= np.linalg.norm(v_mu) + self.eps

        # Gradient-like mean update scaled by normalized reward.
        self.mu += self.lr_mu * r * v_mu

        # Direction for covariance update:
        # push sigma towards outer(diff, diff) (natural policy gradient–like).
        v_sigma = np.outer(diff, diff) - self.sigma

        # Normalize matrix update to prevent excessively large steps.
        v_sigma /= np.linalg.norm(v_sigma) + self.eps

        # Covariance update (symmetric matrix before post-processing).
        self.sigma += self.lr_sigma * r * v_sigma

        # Enforce exact symmetry numerically: sigma = (sigma + sigma.T) / 2.
        self.sigma = 0.5 * (self.sigma + self.sigma.T)

        # Ensure all variances stay above min_var to keep sigma positive
        # definite (or at least well-conditioned).
        diag = np.diag(self.sigma)
        diag = np.maximum(diag, self.min_var)
        np.fill_diagonal(self.sigma, diag)
        return

    def add_dimension(
        self, mu: float | np.ndarray = 0.0, sigma: float | np.ndarray = 100.0
    ) -> None:
        """
        Add one or multiple dimensions to the process.

        Args:
            mu (float | np.ndarray): Mean(s) for the new dimension(s).
            sigma (float | np.ndarray): Std or covariance for the new dimension(s).
        """
        # New mean components.
        new_mu = (
            np.array([mu], dtype=float)
            if np.isscalar(mu)
            else np.asarray(mu, dtype=float).copy()
        )

        # New covariance components.
        new_cov = (
            (sigma**2) * np.eye(new_mu.size, dtype=float)
            if np.isscalar(sigma)
            else np.asarray(sigma, dtype=float).copy()
        )

        # Create enlarged covariance matrix and copy existing structure.
        new_sigma = np.zeros((self.N + new_mu.size, self.N + new_mu.size), dtype=float)
        new_sigma[: self.N, : self.N] = self.sigma
        new_sigma[self.N :, self.N :] = new_cov

        # Update parameters.
        self.mu = np.concatenate([self.mu, new_mu])
        self.sigma = new_sigma
        return

    def remove_dimension(self, idx: int | np.ndarray) -> None:
        """
        Remove one or multiple dimensions from the process.

        Args:
            idx (int | np.ndarray): Index or indices of the dimensions to remove.
        """
        # Normalize indices to a 1D array.
        idx = (
            np.array([idx], dtype=int)
            if isinstance(idx, int)
            else np.asarray(idx, dtype=int).copy()
        )

        # Remove components from mean vector.
        self.mu = np.delete(self.mu, idx, axis=0)

        # Remove corresponding rows and columns from covariance matrix.
        self.sigma = np.delete(self.sigma, idx, axis=0)
        self.sigma = np.delete(self.sigma, idx, axis=1)
        return

    def _process_reward(self, r: np.ndarray) -> np.ndarray:
        """
        Process raw reward by centering and normalizing.

        Args:
            r (np.ndarray): Raw reward value received from environment.

        Returns:
            np.ndarray: Stabilized reward suitable for parameter updates.
        """
        # Exponential moving average for reward baseline (center of rewards).
        self.r_baseline = (1.0 - self.r_alpha) * self.r_baseline + self.r_alpha * r

        # Center reward around the moving baseline.
        r_centered = r - self.r_baseline

        # Exponential moving average of absolute centered reward,
        # used as a scale (like a running standard deviation).
        self.r_scale = (1.0 - self.r_beta) * self.r_scale + self.r_beta * abs(
            r_centered
        )

        # Normalize reward to have roughly unit scale; eps prevents division by 0.
        r_norm = r_centered / (self.r_scale + self.eps)
        return r_norm

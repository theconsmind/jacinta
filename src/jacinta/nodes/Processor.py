from __future__ import annotations

import json

import numpy as np


class Processor:
    """
    Processor maintains and updates a stochastic policy over actions
    represented by a multivariate Gaussian N(mu, Sigma).

    It can:
        - sample actions from the current Gaussian (process_forward)
        - update its parameters using a scalar reward signal (process_backward)

    This makes it suitable as Jacinta's "internal decision maker" or
    search mechanism in continuous spaces.
    """

    def __init__(
        self,
        size: int,
        mu: float | np.ndarray = 0.0,
        sigma: float | np.ndarray = 100.0,
        lr_mu: float = 1e-3,
        lr_sigma: float = 1e-4,
        min_var: float = 1e-8,
        r_alpha: float = 1e-2,
        r_beta: float = 1e-2,
        eps: float = 1e-12,
    ) -> None:
        """
        Initialize a stochastic Processor.

        Args:
            size (int): Dimensionality of the process.
            mu (float | np.ndarray): Initial mean vector or scalar mean.
            sigma (float | np.ndarray): Initial covariance matrix or scalar std.
            lr_mu (float): Learning rate for mean updates.
            lr_sigma (float): Learning rate for covariance updates.
            min_var (float): Minimum variance threshold for stability.
            r_alpha (float): EMA factor for reward baseline update.
            r_beta (float): EMA factor for reward scale update.
            eps (float): Numerical epsilon for stability.
        """
        # Last sampled point from process_forward, needed for credit assignment
        # in process_backward. Set to None when no valid sample is cached.
        self._last_y: np.ndarray | None = None

        # Mean of the Gaussian:
        # - scalar -> broadcast to all dimensions
        # - array  -> copied to avoid external aliasing
        self.mu = (
            np.full(size, mu, dtype=float)
            if isinstance(mu, float)
            else np.asarray(mu, dtype=float).copy()
        )

        # Covariance of the Gaussian:
        # - scalar sigma -> interpreted as std, converted to sigma^2 * I
        # - array        -> used as full covariance matrix (copied)
        self.sigma = (
            (sigma**2) * np.eye(size, dtype=float)
            if isinstance(sigma, float)
            else np.asarray(sigma, dtype=float).copy()
        )

        # Lower bound on per-dimension variance to avoid collapse and
        # numerical issues (e.g. singular covariance matrices).
        self.min_var = min_var
        self.eps = eps

        # Step sizes for gradient-like updates of mean and covariance.
        self.lr_mu = lr_mu
        self.lr_sigma = lr_sigma

        # Reward normalization state:
        # - r_baseline: running mean of rewards
        # - r_scale:    running mean of |reward - baseline|
        self.r_baseline: float = 0.0
        self.r_scale: float = 0.0
        self.r_alpha = r_alpha
        self.r_beta = r_beta
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

        # Copy cached last sample if present (keeps learning continuity).
        if self._last_y is not None:
            processor._last_y = self._last_y.copy()

        # Copy reward normalization state.
        processor.r_baseline = self.r_baseline
        processor.r_scale = self.r_scale
        return processor

    def to_dict(self) -> dict[str, any]:
        """
        Serialize Processor state to a dictionary.

        Returns:
            dict[str, any]: Serializable snapshot of the Processor.
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
            "r_baseline": self.r_baseline,
            "r_scale": self.r_scale,
            "r_alpha": self.r_alpha,
            "r_beta": self.r_beta,
            "eps": self.eps,
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, any]) -> Processor:
        """
        Build a Processor instance from a dict.

        Args:
            data (dict[str, any]): Serialized Processor config.

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
            r_baseline=float(data["r_baseline"]),
            r_scale=float(data["r_scale"]),
            r_alpha=float(data["r_alpha"]),
            r_beta=float(data["r_beta"]),
            eps=float(data["eps"]),
        )
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

    def process_forward(self) -> np.ndarray:
        """
        Sample from the current Gaussian distribution.

        Returns:
            np.ndarray: Sampled vector from N(mu, sigma).
        """
        # Standard normal noise in R^N.
        z = np.random.randn(self.N)

        # Cholesky factorization: sigma = L L^T, assumes sigma is SPD.
        L = np.linalg.cholesky(self.sigma)

        # Reparameterization: y = mu + L z ~ N(mu, sigma).
        y = self.mu + L @ z

        # Cache last sample for use in process_backward().
        self._last_y = y
        return y

    def process_backward(self, r: float) -> None:
        """
        Update mean and covariance based on reward signal.

        Args:
            r (float): Scalar reward used to scale the update.
        """
        # Normalize reward to stabilize learning across different scales.
        r = self._process_reward(r)

        # Deviation of last sample from current mean.
        diff = self._last_y - self.mu

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

        # Enforce exact symmetry numerically: sigma = (sigma + sigma^T) / 2.
        self.sigma = 0.5 * (self.sigma + self.sigma.T)

        # Ensure all variances stay above min_var to keep sigma positive
        # definite (or at least well-conditioned).
        diag = np.diag(self.sigma)
        diag = np.maximum(diag, self.min_var)
        np.fill_diagonal(self.sigma, diag)

        # Invalidate cached sample: only one backward step per forward sample.
        self._last_y = None
        return

    def add_dimension(self, mu: float = 0.0, sigma: float = 100.0) -> None:
        """
        Add a new dimension to the process.

        Args:
            mu (float): Mean for the new dimension.
            sigma (float): Std for the new dimension.
        """
        # Extend mean vector with new component.
        new_mu = np.concatenate([self.mu, np.array([mu], dtype=float)])

        # Create enlarged covariance matrix and copy existing structure.
        new_sigma = np.zeros((self.N + 1, self.N + 1), dtype=float)
        new_sigma[: self.N, : self.N] = self.sigma

        # Initialize new dimension as independent with variance sigma^2.
        new_sigma[self.N, self.N] = sigma**2

        self.mu = new_mu
        self.sigma = new_sigma

        # Any previous sample no longer matches the new dimensionality.
        self._last_y = None
        return

    def remove_dimension(self, idx: int) -> None:
        """
        Remove a specific dimension from the process.

        Args:
            idx (int): Index of the dimension to remove.
        """
        # Remove component from mean vector.
        self.mu = np.delete(self.mu, idx, axis=0)

        # Remove corresponding row and column from covariance matrix
        # to keep it consistent with the reduced dimensionality.
        self.sigma = np.delete(self.sigma, idx, axis=0)
        self.sigma = np.delete(self.sigma, idx, axis=1)

        # Invalidate any cached sample (shape no longer matches).
        self._last_y = None
        return

    def _process_reward(self, r: float) -> float:
        """
        Process raw reward by centering and normalizing.

        Args:
            r (float): Raw reward value received from environment.

        Returns:
            float: Stabilized reward suitable for parameter updates.
        """
        # Exponential moving average for reward baseline (center of rewards).
        self.r_baseline = (
            (1.0 - self.r_alpha) * self.r_baseline + self.r_alpha * r
            if self.r_baseline is not None
            else r
        )

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

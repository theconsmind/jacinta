from __future__ import annotations

import json

import numpy as np


class Processor:

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

        Returns:
            None
        """
        self._last_y: np.ndarray | None = None
        self.mu = (
            np.full(size, mu, dtype=float)
            if isinstance(mu, float)
            else mu.astype(float).copy()
        )
        self.sigma = (
            (sigma**2) * np.eye(size, dtype=float)
            if isinstance(sigma, float)
            else sigma.astype(float).copy()
        )
        self.min_var = min_var
        self.eps = eps
        self.lr_mu = lr_mu
        self.lr_sigma = lr_sigma
        self.r_baseline: float = 0.0
        self.r_scale: float = 0.0
        self.r_alpha = r_alpha
        self.r_beta = r_beta
        return

    @property
    def N(self) -> int:
        """
        Return the dimensionality of the Processor.

        Args:
            None

        Returns:
            int: Number of dimensions.
        """
        N = self.mu.size
        return N

    def copy(self) -> Processor:
        """
        Create a deep copy of the current Processor.

        Args:
            None

        Returns:
            Processor: A new Processor with the same parameters.
        """
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
        if self._last_y is not None:
            processor._last_y = self._last_y.copy()
        processor.r_baseline = self.r_baseline
        processor.r_scale = self.r_scale
        return processor

    def to_dict(self) -> dict[str, any]:
        """
        Serialize Processor state to a dictionary.

        Args:
            None

        Returns:
            dict[str, any]: Serializable snapshot of the Processor.
        """
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

        Returns:
            None
        """
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
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        processor = cls.from_dict(data)
        return processor

    def process_forward(self) -> np.ndarray:
        """
        Sample from the current Gaussian distribution.

        Args:
            None

        Returns:
            np.ndarray: Sampled vector from N(mu, sigma).
        """
        z = np.random.randn(self.N)
        L = np.linalg.cholesky(self.sigma)
        y = self.mu + L @ z
        self._last_y = y
        return y

    def process_backward(self, r: float) -> None:
        """
        Update mean and covariance based on reward signal.

        Args:
            r (float): Scalar reward used to scale the update.

        Returns:
            None
        """
        r = self._process_reward(r)
        diff = self._last_y - self.mu
        try:
            v_mu = np.linalg.solve(self.sigma, diff)
        except np.linalg.LinAlgError:
            v_mu = np.linalg.pinv(self.sigma) @ diff
        v_mu /= np.linalg.norm(v_mu) + self.eps
        self.mu += self.lr_mu * r * v_mu
        v_sigma = np.outer(diff, diff) - self.sigma
        v_sigma /= np.linalg.norm(v_sigma) + self.eps
        self.sigma += self.lr_sigma * r * v_sigma
        self.sigma = 0.5 * (self.sigma + self.sigma.T)
        diag = np.diag(self.sigma)
        diag = np.maximum(diag, self.min_var)
        np.fill_diagonal(self.sigma, diag)
        self._last_y = None
        return

    def add_dimension(self, mu: float = 0.0, sigma: float = 100.0) -> None:
        """
        Add a new dimension to the process.

        Args:
            mu (float): Mean for the new dimension.
            sigma (float): Std for the new dimension.

        Returns:
            None
        """
        new_mu = np.concatenate([self.mu, np.array([mu], dtype=float)])
        new_sigma = np.zeros((self.N + 1, self.N + 1), dtype=float)
        new_sigma[: self.N, : self.N] = self.sigma
        new_sigma[self.N, self.N] = sigma**2
        self.mu = new_mu
        self.sigma = new_sigma
        self._last_y = None
        return

    def remove_dimension(self, idx: int) -> None:
        """
        Remove a specific dimension from the process.

        Args:
            idx (int): Index of the dimension to remove.

        Returns:
            None
        """
        self.mu = np.delete(self.mu, idx, axis=0)
        self.sigma = np.delete(self.sigma, idx, axis=0)
        self.sigma = np.delete(self.sigma, idx, axis=1)
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
        self.r_baseline = (
            (1.0 - self.r_alpha) * self.r_baseline + self.r_alpha * r
            if self.r_baseline is not None
            else r
        )
        r_centered = r - self.r_baseline
        self.r_scale = (1.0 - self.r_beta) * self.r_scale + self.r_beta * abs(
            r_centered
        )
        r_norm = r_centered / (self.r_scale + self.eps)
        return r_norm

from __future__ import annotations

import math
from typing import Any

from .Evaluator import Evaluator


class ZScoreEvaluator(Evaluator):
    """
    An Evaluator that assigns a z-score advantage to a feedback.

    Attributes:
        mean (float | None): The mean of the feedback.
        var (float | None): The variance of the feedback.
        mean_ema_rate (float): The mean EMA rate.
        var_ema_rate (float): The variance EMA rate.
        eps (float): A small positive value used for numerical stability.
    """

    __slots__ = ("_mean", "_var", "_mean_ema_rate", "_var_ema_rate", "_eps")

    def __init__(
        self,
        mean_ema_rate: float,
        var_ema_rate: float,
        eps: float = 1e-9,
    ) -> None:
        """
        Initialize a ZScoreEvaluator.

        Args:
            mean_ema_rate (float): The mean EMA rate.
            var_ema_rate (float): The variance EMA rate.
            eps (float): A small positive value used for numerical stability.
        """
        # mean_ema_rate validations
        if not isinstance(mean_ema_rate, (float, int)):
            raise TypeError("mean_ema_rate must be a float.")
        if not (0.0 <= mean_ema_rate <= 1.0):
            raise ValueError("mean_ema_rate must be in [0, 1].")
        # var_ema_rate validations
        if not isinstance(var_ema_rate, (float, int)):
            raise TypeError("var_ema_rate must be a float.")
        if not (0.0 <= var_ema_rate <= 1.0):
            raise ValueError("var_ema_rate must be in [0, 1].")
        # eps validations
        if not isinstance(eps, (float, int)):
            raise TypeError("eps must be a float.")
        if eps <= 0.0:
            raise ValueError("eps must be greater than 0.0.")
        # initializations
        object.__setattr__(self, "_frozen", False)
        self._mean = None
        self._var = None
        self._mean_ema_rate = float(mean_ema_rate)
        self._var_ema_rate = float(var_ema_rate)
        self._eps = float(eps)
        object.__setattr__(self, "_frozen", True)
        return

    def __repr__(self) -> str:
        """
        Get the representation of the evaluator.

        Returns:
            str: The representation of the evaluator.
        """
        result = (
            f"{self.__class__.__name__}"
            f"(mean={self._mean!r}, var={self._var!r}, "
            f"mean_ema_rate={self._mean_ema_rate!r}, "
            f"var_ema_rate={self._var_ema_rate!r}, "
            f"eps={self._eps!r})"
        )
        return result

    def __call__(self, feedback: float) -> float | None:
        """
        Get the advantage produced by the given feedback.

        Args:
            feedback (float): The feedback.

        Returns:
            float | None: The advantage produced by the given feedback.
        """
        # feedback validations
        if not isinstance(feedback, (float, int)):
            raise TypeError("feedback must be a float.")
        # update statistics
        advantage = None
        feedback = float(feedback)
        # if there is no mean, initialize it
        if self._mean is None:
            object.__setattr__(self, "_frozen", False)
            self._mean = feedback
            object.__setattr__(self, "_frozen", True)
        # if there is no variance, initialize it
        elif self._var is None:
            object.__setattr__(self, "_frozen", False)
            delta = feedback - self._mean
            self._mean += self._mean_ema_rate * delta
            self._var = delta**2
            object.__setattr__(self, "_frozen", True)
        # calculate the z-score advantage and update statistics
        else:
            advantage = (feedback - self._mean) / (math.sqrt(self._var) + self._eps)
            object.__setattr__(self, "_frozen", False)
            delta = feedback - self._mean
            self._mean += self._mean_ema_rate * delta
            self._var = (
                1.0 - self._var_ema_rate
            ) * self._var + self._var_ema_rate * delta**2
            object.__setattr__(self, "_frozen", True)
        return advantage

    @property
    def mean(self) -> float | None:
        """
        Get the mean of the evaluator.

        Returns:
            float | None: The mean of the evaluator.
        """
        return self._mean

    @property
    def var(self) -> float | None:
        """
        Get the variance of the evaluator.

        Returns:
            float | None: The variance of the evaluator.
        """
        return self._var

    @property
    def mean_ema_rate(self) -> float:
        """
        Get the mean EMA rate of the evaluator.

        Returns:
            float: The mean EMA rate of the evaluator.
        """
        return self._mean_ema_rate

    @property
    def var_ema_rate(self) -> float:
        """
        Get the variance EMA rate of the evaluator.

        Returns:
            float: The variance EMA rate of the evaluator.
        """
        return self._var_ema_rate

    @property
    def eps(self) -> float:
        """
        Get the epsilon of the evaluator.

        Returns:
            float: The epsilon of the evaluator.
        """
        return self._eps

    def __eq__(self, other: object) -> bool:
        """
        Check if two evaluators are equal.

        Args:
            other (object): The object to compare with.

        Returns:
            bool: True if the evaluators are equal, False otherwise.
        """
        # other validations
        if type(self) is not type(other):
            return NotImplemented
        # equality check
        result = (
            self._mean == other._mean
            and self._var == other._var
            and self._mean_ema_rate == other._mean_ema_rate
            and self._var_ema_rate == other._var_ema_rate
            and self._eps == other._eps
        )
        return result

    def to_dict(self) -> dict[str, Any]:
        """
        Get the dictionary representation of the evaluator.

        Returns:
            dict[str, Any]: The dictionary representation of the evaluator.
        """
        result = {
            "type": self.__class__.__name__,
            "mean": self._mean,
            "var": self._var,
            "mean_ema_rate": self._mean_ema_rate,
            "var_ema_rate": self._var_ema_rate,
            "eps": self._eps,
        }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ZScoreEvaluator:
        """
        Create an evaluator from a dictionary.

        Args:
            data (dict[str, Any]): The dictionary representation of the evaluator.

        Returns:
            ZScoreEvaluator: The evaluator.
        """
        # data validations
        if not isinstance(data, dict):
            raise TypeError("data must be a dict.")
        if "type" not in data:
            raise KeyError("data must contain the key 'type'.")
        if data["type"] != cls.__name__:
            raise ValueError(f"data['type'] must be a {cls.__name__}.")
        if "mean" not in data:
            raise KeyError("data must contain the key 'mean'.")
        if data["mean"] is not None and not isinstance(data["mean"], (float, int)):
            raise TypeError("data['mean'] must be a float.")
        if "var" not in data:
            raise KeyError("data must contain the key 'var'.")
        if data["var"] is not None and not isinstance(data["var"], (float, int)):
            raise TypeError("data['var'] must be a float.")
        if "mean_ema_rate" not in data:
            raise KeyError("data must contain the key 'mean_ema_rate'.")
        if "var_ema_rate" not in data:
            raise KeyError("data must contain the key 'var_ema_rate'.")
        if "eps" not in data:
            raise KeyError("data must contain the key 'eps'.")
        # initializations
        result = cls(
            data["mean_ema_rate"],
            data["var_ema_rate"],
            data["eps"],
        )
        # update mean and var
        object.__setattr__(result, "_frozen", False)
        result._mean = float(data["mean"]) if data["mean"] is not None else None
        result._var = float(data["var"]) if data["var"] is not None else None
        object.__setattr__(result, "_frozen", True)
        return result

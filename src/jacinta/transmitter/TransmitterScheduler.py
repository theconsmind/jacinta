from __future__ import annotations


class TransmitterScheduler:
    """
    The TransmitterScheduler defines a depth-based schedule for a parameter
    that transitions from an initial value to a final value over a fixed number
    of levels.

    Attributes:
        start (float): The initial value of the parameter.
        end (float): The final value of the parameter.
        steps (int): The number of levels over which the parameter transitions.
    """

    __slots__ = ("_start", "_end", "_steps")

    def __init__(self, start: float, end: float, steps: int) -> None:
        """
        Initialize the TransmitterScheduler.

        Args:
            start (float): The initial value of the parameter.
            end (float): The final value of the parameter.
            steps (int): The number of levels over which the parameter transitions.
        """
        # start validations
        if not isinstance(start, (float, int)):
            raise TypeError("start must be a float.")
        if start <= 0:
            raise ValueError("start must be greater than 0.")
        # end validations
        if not isinstance(end, (float, int)):
            raise TypeError("end must be a float.")
        if end <= 0:
            raise ValueError("end must be greater than 0.")
        # steps validations
        if not isinstance(steps, int):
            raise TypeError("steps must be an int.")
        if steps <= 0:
            raise ValueError("steps must be greater than 0.")
        # initializations
        self._start = float(start)
        self._end = float(end)
        self._steps = steps
        return

    @property
    def start(self) -> float:
        """
        Get the initial value of the parameter.

        Returns:
            float: The initial value of the parameter.
        """
        return self._start

    @property
    def end(self) -> float:
        """
        Get the final value of the parameter.

        Returns:
            float: The final value of the parameter.
        """
        return self._end

    @property
    def steps(self) -> int:
        """
        Get the number of levels over which the parameter transitions.

        Returns:
            int: The number of levels over which the parameter transitions.
        """
        return self._steps

    def value(self, depth: int) -> float:
        """
        Get the parameter value for a given depth.

        Args:
            depth (int): The depth in the Transmitter tree.

        Returns:
            float: The value of the parameter for the given depth.
        """
        # depth validations
        if not isinstance(depth, int):
            raise TypeError("depth must be an int.")
        if depth < 0:
            raise ValueError("depth must be greater than or equal to 0.")
        # compute the value based on the depth
        depth = depth if depth <= self._steps else self._steps
        value = self._start * (self._end / self._start) ** (depth / self._steps)
        return value

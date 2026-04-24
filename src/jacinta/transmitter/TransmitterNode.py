from __future__ import annotations


class TransmitterNode:
    """
    A TransmitterNode represents a node in the Transmitter tree.

    Attributes:
        left (float): The left bound of the interval.
        right (float): The right bound of the interval.
        parent_id (int): The ID of the parent node.
        left_child_id (int): The ID of the left child node.
        right_child_id (int): The ID of the right child node.
        weight (float): The weight of the node.
        hits_left (int): The number of hits left to split the node.
        mass (float): The probability of the node being selected.
        depth (int): The depth of the node.
        learning_rate (float): The learning rate of the node.
    """

    __slots__ = (
        "_left",
        "_right",
        "_parent_id",
        "_left_child_id",
        "_right_child_id",
        "_weight",
        "_hits_left",
        "_mass",
        "_depth",
        "_learning_rate",
    )

    def __init__(
        self,
        left: float,
        right: float,
        parent_id: int,
        left_child_id: int,
        right_child_id: int,
        weight: float,
        hits_left: int,
        mass: float,
        depth: int,
        learning_rate: float,
    ) -> None:
        """
        Initialize the TransmitterNode.

        Args:
            left (float): The left bound of the interval.
            right (float): The right bound of the interval.
            parent_id (int): The ID of the parent node.
            left_child_id (int): The ID of the left child node.
            right_child_id (int): The ID of the right child node.
            weight (float): The weight of the node.
            hits_left (int): The number of hits left to split the node.
            mass (float): The probability of the node being selected.
            depth (int): The depth of the node.
            learning_rate (float): The learning rate of the node.
        """
        # left & right validations
        if not isinstance(left, (float, int)):
            raise TypeError("left must be a float.")
        if not isinstance(right, (float, int)):
            raise TypeError("right must be a float.")
        if left >= right:
            raise ValueError("left must be lower than right.")
        # node_ids validations
        if not isinstance(parent_id, int):
            raise TypeError("parent_id must be an int.")
        if parent_id < -1:
            raise ValueError("parent_id must be greater than or equal to -1.")
        if not isinstance(left_child_id, int):
            raise TypeError("left_child_id must be an int.")
        if left_child_id < -1:
            raise ValueError("left_child_id must be greater than or equal to -1.")
        if not isinstance(right_child_id, int):
            raise TypeError("right_child_id must be an int.")
        if right_child_id < -1:
            raise ValueError("right_child_id must be greater than or equal to -1.")
        # weight validations
        if not isinstance(weight, (float, int)):
            raise TypeError("weight must be a float.")
        if weight <= 0:
            raise ValueError("weight must be greater than 0.")
        # hits_left validations
        if not isinstance(hits_left, int):
            raise TypeError("hits_left must be an int.")
        if hits_left < 0:
            raise ValueError("hits_left must be greater than or equal to 0.")
        # mass validations
        if not isinstance(mass, (float, int)):
            raise TypeError("mass must be a float.")
        if mass <= 0:
            raise ValueError("mass must be greater than 0.")
        # depth validations
        if not isinstance(depth, int):
            raise TypeError("depth must be an int.")
        if depth < 0:
            raise ValueError("depth must be greater than or equal to 0.")
        # learning_rate validations
        if not isinstance(learning_rate, (float, int)):
            raise TypeError("learning_rate must be a float.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0.")
        # initializations
        self._left = float(left)
        self._right = float(right)
        self._parent_id = parent_id
        self._left_child_id = left_child_id
        self._right_child_id = right_child_id
        self._weight = float(weight)
        self._hits_left = hits_left
        self._mass = float(mass)
        self._depth = depth
        self._learning_rate = float(learning_rate)
        return

    @property
    def left(self) -> float:
        """
        Get the left bound of the interval.

        Returns:
            float: The left bound of the interval.
        """
        return self._left

    @property
    def right(self) -> float:
        """
        Get the right bound of the interval.

        Returns:
            float: The right bound of the interval.
        """
        return self._right

    @property
    def parent_id(self) -> int:
        """
        Get the ID of the parent node.

        Returns:
            int: The ID of the parent node.
        """
        return self._parent_id

    @property
    def left_child_id(self) -> int:
        """
        Get the ID of the left child node.

        Returns:
            int: The ID of the left child node.
        """
        return self._left_child_id

    @left_child_id.setter
    def left_child_id(self, value: int) -> None:
        """
        Set the ID of the left child node.

        Args:
            value (int): The ID of the left child node.
        """
        if not isinstance(value, int):
            raise TypeError("left_child_id must be an int.")
        if value < -1:
            raise ValueError("left_child_id must be greater than or equal to -1.")
        self._left_child_id = value
        return

    @property
    def right_child_id(self) -> int:
        """
        Get the ID of the right child node.

        Returns:
            int: The ID of the right child node.
        """
        return self._right_child_id

    @right_child_id.setter
    def right_child_id(self, value: int) -> None:
        """
        Set the ID of the right child node.

        Args:
            value (int): The ID of the right child node.
        """
        if not isinstance(value, int):
            raise TypeError("right_child_id must be an int.")
        if value < -1:
            raise ValueError("right_child_id must be greater than or equal to -1.")
        self._right_child_id = value
        return

    @property
    def weight(self) -> float:
        """
        Get the weight of the node.

        Returns:
            float: The weight of the node.
        """
        return self._weight

    @weight.setter
    def weight(self, value: float) -> None:
        """
        Set the weight of the node.

        Args:
            value (float): The weight of the node.
        """
        if not isinstance(value, (float, int)):
            raise TypeError("weight must be a float.")
        if value <= 0:
            raise ValueError("weight must be greater than 0.")
        self._weight = float(value)
        return

    @property
    def hits_left(self) -> int:
        """
        Get the number of hits left to split the node.

        Returns:
            int: The number of hits left to split the node.
        """
        return self._hits_left

    @hits_left.setter
    def hits_left(self, value: int) -> None:
        """
        Set the number of hits left to split the node.

        Args:
            value (int): The number of hits left to split the node.
        """
        if not isinstance(value, int):
            raise TypeError("hits_left must be an int.")
        if value < 0:
            raise ValueError("hits_left must be greater than or equal to 0.")
        self._hits_left = value
        return

    @property
    def mass(self) -> float:
        """
        Get the probability of the node being selected.

        Returns:
            float: The probability of the node being selected.
        """
        return self._mass

    @mass.setter
    def mass(self, value: float) -> None:
        """
        Set the probability of the node being selected.

        Args:
            value (float): The probability of the node being selected.
        """
        if not isinstance(value, (float, int)):
            raise TypeError("mass must be a float.")
        if value <= 0:
            raise ValueError("mass must be greater than 0.")
        self._mass = float(value)
        return

    @property
    def depth(self) -> int:
        """
        Get the depth of the node.

        Returns:
            int: The depth of the node.
        """
        return self._depth

    @property
    def learning_rate(self) -> float:
        """
        Get the learning rate of the node.

        Returns:
            float: The learning rate of the node.
        """
        return self._learning_rate

    @property
    def is_leaf(self) -> bool:
        """
        Check if the node is a leaf.

        Returns:
            bool: True if the node is a leaf, False otherwise.
        """
        result = self._left_child_id == -1 and self._right_child_id == -1
        return result

    @property
    def length(self) -> float:
        """
        Get the length of the interval.

        Returns:
            float: The length of the interval.
        """
        result = self._right - self._left
        return result

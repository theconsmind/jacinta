from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .Transmitter import Transmitter
from .TransmitterSample import TransmitterSample


class TransmitterTree:
    """
    A TransmitterTree is a tree of Transmitters with irregular shape.

    Attributes:
        transmitters (tuple[Transmitter | tuple]): The transmitters.
        shape (tuple[int | tuple]): The shape of the tree.
    """

    __slots__ = ("_transmitters", "_shape", "_frozen")

    def __init__(self, transmitters: tuple[Transmitter | tuple]) -> None:
        """
        Initialize a TransmitterTree.

        Args:
            transmitters (tuple[Transmitter | tuple]): The transmitters.
        """
        # transmitters validations
        if not isinstance(transmitters, (tuple, list)):
            raise TypeError("transmitters must be a tuple.")
        if not self._validate_structure(transmitters):
            raise TypeError(
                "All transmitters must be a Transmitter or a tuple of Transmitters."
            )
        # initializations
        super().__setattr__("_frozen", False)
        self._transmitters = self._parse_structure(transmitters)
        self._shape = self._build_shape(self._transmitters, path=())
        super().__setattr__("_frozen", True)
        return

    def __repr__(self) -> str:
        """
        Get the representation of the tree.

        Returns:
            str: The representation of the tree.
        """
        result = f"{self.__class__.__name__}({self._transmitters!r})"
        return result

    @property
    def shape(self) -> tuple[int | tuple]:
        """
        Get the shape of the tree.

        Returns:
            tuple[int | tuple]: The shape of the tree.
        """
        return self._shape

    def __eq__(self, other: object) -> bool:
        """
        Check if two TransmitterTrees are equal.

        Args:
            other (object): The object to compare with.

        Returns:
            bool: True if the trees are equal, False otherwise.
        """
        # type validations
        if not isinstance(other, TransmitterTree):
            return NotImplemented
        # equality check
        result = self._transmitters == other._transmitters
        return result

    def __getitem__(self, key: int | tuple[int]) -> Transmitter | TransmitterTree:
        """
        Get the transmitter or subtree at the given path.

        Args:
            key (int | tuple[int]): The path to the transmitter or subtree.

        Returns:
            Transmitter | TransmitterTree: The transmitter or subtree at the given path.
        """
        # key validations
        if not isinstance(key, (int, tuple)):
            raise TypeError("key must be an int or a tuple.")
        if isinstance(key, tuple):
            for idx in key:
                if not isinstance(idx, int):
                    raise TypeError("All keys must be ints.")
        # handle integer keys
        if isinstance(key, int):
            try:
                result = self._transmitters[key]
            except IndexError as exc:
                raise IndexError("Invalid path to the TransmitterTree.") from exc
        # handle tuple keys
        elif isinstance(key, tuple):
            result = self._transmitters
            for idx in key:
                if not isinstance(result, tuple):
                    raise IndexError("Invalid path to the TransmitterTree.")
                try:
                    result = result[idx]
                except IndexError as exc:
                    raise IndexError("Invalid path to the TransmitterTree.") from exc
        # tuple results must be wrapped as TransmitterTrees
        if isinstance(result, tuple):
            result = TransmitterTree(result)
        return result

    def forward(self, bias: float = 0.0) -> tuple[TransmitterSample | tuple]:
        """
        Generate samples using the tree.

        Args:
            bias (float): The bias to use for the generation.
                Defaults to 0.0.

        Returns:
            tuple[TransmitterSample | tuple]: The generated samples.
        """
        # bias validations
        if not isinstance(bias, (float, int)):
            raise TypeError("bias must be a float")
        if not (-1.0 <= bias <= 1.0):
            raise ValueError("bias must be in [-1, 1]")
        # generate samples
        result = self._forward_structure(self._transmitters, float(bias))
        return result

    def backward(
        self,
        samples: tuple[TransmitterSample | tuple],
        feedback: float,
    ) -> None:
        """
        Update the tree based on the feedback.

        Args:
            samples (tuple[TransmitterSample | tuple]): The samples to use
                for the update.
            feedback (float): The feedback associated to the samples.
        """
        # samples validations
        if not self._validate_samples(self._transmitters, samples):
            raise TypeError("samples is not a valid sample structure for the tree.")
        # feedback validations
        if not isinstance(feedback, (float, int)):
            raise TypeError("feedback must be a float")
        if not (-1.0 <= feedback <= 1.0):
            raise ValueError("feedback must be in [-1, 1]")
        # backpropagate the feedback
        self._backward_structure(self._transmitters, samples, float(feedback))
        return

    def to_dict(self) -> dict[str, Any]:
        """
        Get the dictionary representation of the tree.

        Returns:
            dict[str, Any]: The dictionary representation of the tree.
        """
        result = {
            "type": self.__class__.__name__,
            "transmitters": self._serialize_structure(self._transmitters),
        }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransmitterTree:
        """
        Create a TransmitterTree from a dictionary.

        Args:
            data (dict[str, Any]): The dictionary representation of the tree.

        Returns:
            TransmitterTree: The TransmitterTree instance.
        """
        # data validations
        if not isinstance(data, dict):
            raise TypeError("data must be a dict.")
        if "type" not in data:
            raise KeyError("data must contain the key 'type'.")
        if data["type"] != cls.__name__:
            raise ValueError(f"data['type'] must be a {cls.__name__}.")
        if "transmitters" not in data:
            raise KeyError("data must contain the key 'transmitters'.")
        # initializations
        result = cls(cls._deserialize_structure(data["transmitters"]))
        return result

    def save(self, path: str | Path, overwrite: bool = False) -> None:
        """
        Save the tree to a json file.

        Args:
            path (str | Path): The path to the file.
            overwrite (bool): Whether to overwrite the file if it exists.
        """
        # path validations
        if not isinstance(path, (str, Path)):
            raise TypeError("path must be a string or a Path.")
        # file validations
        path = Path(path)
        if path.suffix != ".json":
            raise ValueError("path must have a .json extension.")
        if not overwrite and path.exists():
            raise FileExistsError(f"path already exists: {path}")
        # file creation
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=4)
        return

    @classmethod
    def load(cls, path: str | Path) -> TransmitterTree:
        """
        Load the tree from a json file.

        Args:
            path (str | Path): The path to the file.

        Returns:
            TransmitterTree: The TransmitterTree instance.
        """
        # path validations
        if not isinstance(path, (str, Path)):
            raise TypeError("path must be a string or a Path.")
        # file validations
        path = Path(path)
        if path.suffix != ".json":
            raise ValueError("path must have a .json extension.")
        if not path.exists():
            raise FileNotFoundError(f"path does not exist: {path}")
        # file loading
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        result = cls.from_dict(data)
        return result

    @classmethod
    def _validate_structure(
        cls,
        structure: Transmitter | tuple[Transmitter | tuple],
    ) -> bool:
        """
        Validate the structure of the tree.

        Args:
            structure (Transmitter | tuple[Transmitter | tuple]): The structure
                to validate.

        Returns:
            bool: True if the structure is valid, False otherwise.
        """
        is_valid = False
        # check if the structure is a Transmitter
        if isinstance(structure, Transmitter):
            is_valid = True
        # check if the structure is a tuple or list
        elif isinstance(structure, (tuple, list)):
            is_valid = all(cls._validate_structure(item) for item in structure)
        return is_valid

    @classmethod
    def _parse_structure(
        cls,
        structure: Transmitter | tuple[Transmitter | tuple],
    ) -> Transmitter | tuple[Transmitter | tuple]:
        """
        Parse the structure of the tree.

        Args:
            structure (Transmitter | tuple[Transmitter | tuple]): The structure
                to parse.

        Returns:
            Transmitter | tuple[Transmitter | tuple]: The parsed structure.
        """
        if isinstance(structure, (tuple, list)):
            structure = tuple(cls._parse_structure(item) for item in structure)
        return structure

    @classmethod
    def _build_shape(
        cls,
        structure: Transmitter | tuple[Transmitter | tuple],
        path: tuple[int | tuple],
    ) -> tuple[int | tuple]:
        """
        Build the shape of the tree.

        Args:
            structure (Transmitter | tuple[Transmitter | tuple]): The structure
                to build the shape from.
            path (tuple[int | tuple]): The current path from the root to the
                current node.

        Returns:
            tuple[int | tuple]: The shape from the current node to the leaves.
        """
        if isinstance(structure, tuple):
            path = tuple(
                cls._build_shape(item, path + (idx,))
                for idx, item in enumerate(structure)
            )
        return path

    @classmethod
    def _forward_structure(
        cls,
        structure: Transmitter | tuple[Transmitter | tuple],
        bias: float = 0.0,
    ) -> TransmitterSample | tuple[TransmitterSample | tuple]:
        """
        Perform a forward pass through the tree.

        Args:
            structure (Transmitter | tuple[Transmitter | tuple]): The structure
                to perform the forward pass through.
            bias (float): The bias to use for the forward pass.
                Defaults to 0.0.

        Returns:
            TransmitterSample | tuple[TransmitterSample | tuple]: The samples
                from the forward pass.
        """
        if isinstance(structure, Transmitter):
            result = structure.forward(bias)
        elif isinstance(structure, tuple):
            result = tuple(cls._forward_structure(item, bias) for item in structure)
        return result

    @classmethod
    def _validate_samples(
        cls,
        structure: Transmitter | tuple[Transmitter | tuple],
        samples: TransmitterSample | tuple[TransmitterSample | tuple],
    ) -> bool:
        """
        Validate the samples.

        Args:
            structure (Transmitter | tuple[Transmitter | tuple]): The structure
                that generated the samples.
            samples (TransmitterSample | tuple[TransmitterSample | tuple]): The
                samples to validate.

        Returns:
            bool: True if the samples are valid, False otherwise.
        """
        is_valid = False
        if isinstance(structure, Transmitter) and isinstance(
            samples, TransmitterSample
        ):
            is_valid = True
        elif isinstance(structure, tuple) and isinstance(samples, tuple):
            if len(structure) == len(samples):
                is_valid = all(
                    cls._validate_samples(item, sample)
                    for item, sample in zip(structure, samples, strict=True)
                )
        return is_valid

    @classmethod
    def _backward_structure(
        cls,
        structure: Transmitter | tuple[Transmitter | tuple],
        samples: TransmitterSample | tuple[TransmitterSample | tuple],
        feedback: float,
    ) -> None:
        """
        Perform a backward pass through the tree.

        Args:
            structure (Transmitter | tuple[Transmitter | tuple]): The structure
                to perform the backward pass through.
            samples (TransmitterSample | tuple[TransmitterSample | tuple]): The
                samples from the forward pass.
            feedback (float): The feedback associated to the samples.
        """
        if isinstance(structure, Transmitter):
            structure.backward(samples, feedback)
        elif isinstance(structure, tuple):
            for item, sample in zip(structure, samples, strict=True):
                cls._backward_structure(item, sample, feedback)
        return

    @classmethod
    def _serialize_structure(
        cls,
        structure: Transmitter | tuple[Transmitter | tuple],
    ) -> dict[str, Any] | tuple[dict | tuple]:
        """
        Serialize the structure of the tree.

        Args:
            structure (Transmitter | tuple[Transmitter | tuple]): The structure
                to serialize.

        Returns:
            dict[str, Any] | tuple[dict | tuple]: The serialized structure.
        """
        if isinstance(structure, Transmitter):
            result = structure.to_dict()
        elif isinstance(structure, tuple):
            result = tuple(cls._serialize_structure(item) for item in structure)
        return result

    @classmethod
    def _deserialize_structure(
        cls,
        structure: dict[str, Any] | tuple[dict | tuple],
    ) -> Transmitter | tuple[Transmitter | tuple]:
        """
        Deserialize the structure of the tree.

        Args:
            structure (dict[str, Any] | tuple[dict | tuple]): The structure
                to deserialize.

        Returns:
            Transmitter | tuple[Transmitter | tuple]: The deserialized structure.
        """
        if isinstance(structure, dict):
            result = Transmitter.from_dict(structure)
        elif isinstance(structure, (tuple, list)):
            result = tuple(cls._deserialize_structure(item) for item in structure)
        return result

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Set an attribute of the array.

        Args:
            name (str): The name of the attribute.
            value (Any): The value of the attribute.
        """
        # freeze check
        if getattr(self, "_frozen", False):
            raise AttributeError(f"{self.__class__.__name__} is immutable")
        # set the attribute
        super().__setattr__(name, value)
        return

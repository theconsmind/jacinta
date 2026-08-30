from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..processor import Processor
from .node import Node


class Network:
    """ """

    __slots__ = (
        "_nodes",
        "_connections",
        "_rports",
        "_tports",
        "_frozen",
    )

    def __init__(self) -> None:
        """ """
        # initializations
        object.__setattr__(self, "_frozen", False)
        self._nodes = ()
        self._connections = ()
        self._rports = ()
        self._tports = ()
        object.__setattr__(self, "_frozen", True)
        return

    def __repr__(self) -> str:
        """ """
        result = (
            f"{self.__class__.__name__}"
            f"(nodes={self._nodes!r}, connections={self._connections!r}, "
            f"rports={self._rports!r}, tports={self._tports!r})"
        )
        return result

    @property
    def nodes(self) -> tuple[Node, ...]:
        """ """
        return self._nodes

    @property
    def connections(self) -> tuple[tuple[tuple[Node, int], tuple[Node, int]], ...]:
        """ """
        return self._connections

    @property
    def rports(self) -> tuple[tuple[Node, int], ...]:
        """ """
        return self._rports

    @property
    def tports(self) -> tuple[tuple[Node, int], ...]:
        """ """
        return self._tports

    def __eq__(self, other: object) -> bool:
        """ """
        # other validations
        if type(self) is not type(other):
            return NotImplemented
        # equality check
        result = self._has_same_topology(other)
        return result

    def __contains__(self, other: object) -> bool:
        """ """
        # other validations
        if not isinstance(other, Node):
            raise TypeError("other must be a Node.")
        # check if the node is within the network
        result = any(node is other for node in self._nodes)
        return result

    def add_node(self, processor: Processor, defaults: tuple[float, ...]) -> Node:
        """ """
        node = Node(processor, defaults)
        object.__setattr__(self, "_frozen", False)
        self._nodes += (node,)
        object.__setattr__(self, "_frozen", True)
        return node

    def remove_node(self, node: Node) -> None:
        """ """
        # node validations
        if not isinstance(node, Node):
            raise TypeError("node must be a Node.")
        if node not in self:
            raise ValueError("node must be contained in self.")
        # remove node
        object.__setattr__(self, "_frozen", False)
        self._nodes = tuple(
            network_node for network_node in self._nodes if network_node is not node
        )
        # remove connections
        self._connections = tuple(
            connection
            for connection in self._connections
            if connection[0][0] is not node and connection[1][0] is not node
        )
        # remove ports
        self._rports = tuple(port for port in self._rports if port[0] is not node)
        self._tports = tuple(port for port in self._tports if port[0] is not node)
        object.__setattr__(self, "_frozen", True)
        return

    def connect(
        self,
        source: Node,
        source_dim: int,
        target: Node,
        target_dim: int,
    ) -> None:
        """ """
        # source validations
        if not isinstance(source, Node):
            raise TypeError("source must be a Node.")
        if source not in self:
            raise ValueError("source must be contained in self.")
        # source_dim validations
        if not isinstance(source_dim, int):
            raise TypeError("source_dim must be an int.")
        if not (0 <= source_dim < source.tnd):
            raise IndexError("source_dim must be in range.")
        # target validations
        if not isinstance(target, Node):
            raise TypeError("target must be a Node.")
        if target not in self:
            raise ValueError("target must be contained in self.")
        # target_dim validations
        if not isinstance(target_dim, int):
            raise TypeError("target_dim must be an int.")
        if not (0 <= target_dim < target.rnd):
            raise IndexError("target_dim must be in range.")
        if any(
            connection[1][0] is target and connection[1][1] == target_dim
            for connection in self._connections
        ):
            raise ValueError("target_dim is already connected.")
        if any(port[0] is target and port[1] == target_dim for port in self._rports):
            raise ValueError("target_dim is already an rport.")
        # add connection
        connection = ((source, source_dim), (target, target_dim))
        object.__setattr__(self, "_frozen", False)
        self._connections += (connection,)
        object.__setattr__(self, "_frozen", True)
        return

    def disconnect(
        self,
        source: Node,
        source_dim: int,
        target: Node,
        target_dim: int,
    ) -> None:
        """ """
        # source validations
        if not isinstance(source, Node):
            raise TypeError("source must be a Node.")
        if source not in self:
            raise ValueError("source must be contained in self.")
        # source_dim validations
        if not isinstance(source_dim, int):
            raise TypeError("source_dim must be an int.")
        if not (0 <= source_dim < source.tnd):
            raise IndexError("source_dim must be in range.")
        # target validations
        if not isinstance(target, Node):
            raise TypeError("target must be a Node.")
        if target not in self:
            raise ValueError("target must be contained in self.")
        # target_dim validations
        if not isinstance(target_dim, int):
            raise TypeError("target_dim must be an int.")
        if not (0 <= target_dim < target.rnd):
            raise IndexError("target_dim must be in range.")
        # find connection
        connection = next(
            (
                connection
                for connection in self._connections
                if connection[0][0] is source
                and connection[0][1] == source_dim
                and connection[1][0] is target
                and connection[1][1] == target_dim
            ),
            None,
        )
        if connection is None:
            raise ValueError("source_dim and target_dim are not connected.")
        # remove connection
        object.__setattr__(self, "_frozen", False)
        self._connections = tuple(
            network_connection
            for network_connection in self._connections
            if network_connection is not connection
        )
        object.__setattr__(self, "_frozen", True)
        return

    def add_rport(self, node: Node, dim: int) -> None:
        """ """
        # node validations
        if not isinstance(node, Node):
            raise TypeError("node must be a Node.")
        if node not in self:
            raise ValueError("node must be contained in self.")
        # dim validations
        if not isinstance(dim, int):
            raise TypeError("dim must be an int.")
        if not (0 <= dim < node.rnd):
            raise IndexError("dim must be in range.")
        if any(
            connection[1][0] is node and connection[1][1] == dim
            for connection in self._connections
        ):
            raise ValueError("dim is already connected.")
        if any(port[0] is node and port[1] == dim for port in self._rports):
            raise ValueError("dim is already an rport.")
        # add rport
        object.__setattr__(self, "_frozen", False)
        self._rports += ((node, dim),)
        object.__setattr__(self, "_frozen", True)
        return

    def remove_rport(self, node: Node, dim: int) -> None:
        """ """
        # node validations
        if not isinstance(node, Node):
            raise TypeError("node must be a Node.")
        if node not in self:
            raise ValueError("node must be contained in self.")
        # dim validations
        if not isinstance(dim, int):
            raise TypeError("dim must be an int.")
        if not (0 <= dim < node.rnd):
            raise IndexError("dim must be in range.")
        # find rport
        port = next(
            (port for port in self._rports if port[0] is node and port[1] == dim),
            None,
        )
        if port is None:
            raise ValueError("dim is not an rport.")
        # remove rport
        object.__setattr__(self, "_frozen", False)
        self._rports = tuple(rport for rport in self._rports if rport is not port)
        object.__setattr__(self, "_frozen", True)
        return

    def add_tport(self, node: Node, dim: int) -> None:
        """ """
        # node validations
        if not isinstance(node, Node):
            raise TypeError("node must be a Node.")
        if node not in self:
            raise ValueError("node must be contained in self.")
        # dim validations
        if not isinstance(dim, int):
            raise TypeError("dim must be an int.")
        if not (0 <= dim < node.tnd):
            raise IndexError("dim must be in range.")
        if any(port[0] is node and port[1] == dim for port in self._tports):
            raise ValueError("dim is already a tport.")
        # add tport
        object.__setattr__(self, "_frozen", False)
        self._tports += ((node, dim),)
        object.__setattr__(self, "_frozen", True)
        return

    def remove_tport(self, node: Node, dim: int) -> None:
        """ """
        # node validations
        if not isinstance(node, Node):
            raise TypeError("node must be a Node.")
        if node not in self:
            raise ValueError("node must be contained in self.")
        # dim validations
        if not isinstance(dim, int):
            raise TypeError("dim must be an int.")
        if not (0 <= dim < node.tnd):
            raise IndexError("dim must be in range.")
        # find tport
        port = next(
            (port for port in self._tports if port[0] is node and port[1] == dim),
            None,
        )
        if port is None:
            raise ValueError("dim is not a tport.")
        # remove tport
        object.__setattr__(self, "_frozen", False)
        self._tports = tuple(tport for tport in self._tports if tport is not port)
        object.__setattr__(self, "_frozen", True)
        return

    def add_rdimensions(
        self,
        node: Node,
        bounds: tuple[tuple[float, float], ...],
        defaults: tuple[float, ...],
    ) -> None:
        """ """
        # node validations
        if not isinstance(node, Node):
            raise TypeError("node must be a Node.")
        if node not in self:
            raise ValueError("node must be contained in self.")
        # bounds validations
        if not isinstance(bounds, (tuple, list)):
            raise TypeError("bounds must be a tuple.")
        for bound in bounds:
            if not isinstance(bound, (tuple, list)):
                raise TypeError("All bounds must be tuples.")
            if len(bound) != 2:
                raise ValueError("All bounds must have length 2.")
            if not isinstance(bound[0], (float, int)):
                raise TypeError("All lower bounds must be floats.")
            if not isinstance(bound[1], (float, int)):
                raise TypeError("All upper bounds must be floats.")
        # defaults validations
        if not isinstance(defaults, (tuple, list)):
            raise TypeError("defaults must be a tuple.")
        if len(defaults) != len(bounds):
            raise ValueError(f"defaults must have length {len(bounds)}.")
        for default, (lower, upper) in zip(defaults, bounds, strict=True):
            if not isinstance(default, (float, int)):
                raise TypeError("All defaults must be floats.")
            if not (lower <= default < upper):
                raise ValueError("All defaults must be contained in bounds.")
        # add new rbounds to the processor
        node.processor.add_rdimensions(bounds)
        object.__setattr__(node, "_frozen", False)
        node._defaults += tuple(float(default) for default in defaults)
        object.__setattr__(node, "_frozen", True)
        return

    def remove_rdimensions(self, node: Node, dims: set[int]) -> None:
        """ """
        # node validations
        if not isinstance(node, Node):
            raise TypeError("node must be a Node.")
        if node not in self:
            raise ValueError("node must be contained in self.")
        # dims validations
        if not isinstance(dims, (set, tuple, list)):
            raise TypeError("dims must be a set.")
        if len(set(dims)) != len(dims):
            raise ValueError("All dims must be unique.")
        for dim in dims:
            if not isinstance(dim, int):
                raise TypeError("All dims must be ints.")
            if not (0 <= dim < node.rnd):
                raise IndexError("All dims must be in range.")
        # remove rdimensions from the processor
        dims = set(dims)
        node.processor.remove_rdimensions(dims)

        def _remap(old_dim: int) -> int:
            """ """
            new_dim = old_dim - sum(dim < old_dim for dim in dims)
            return new_dim

        # remove defaults
        defaults = tuple(
            default for dim, default in enumerate(node.defaults) if dim not in dims
        )
        object.__setattr__(node, "_frozen", False)
        node._defaults = defaults
        object.__setattr__(node, "_frozen", True)
        # remove connections
        connections = []
        for connection in self._connections:
            (source, source_dim), (target, target_dim) = connection
            if target is node:
                if target_dim in dims:
                    continue
                target_dim = _remap(target_dim)
            connections.append(((source, source_dim), (target, target_dim)))
        # remove rports
        rports = []
        for port_node, dim in self._rports:
            if port_node is node:
                if dim in dims:
                    continue
                dim = _remap(dim)
            rports.append((port_node, dim))
        # update network
        object.__setattr__(self, "_frozen", False)
        self._connections = tuple(connections)
        self._rports = tuple(rports)
        object.__setattr__(self, "_frozen", True)
        return

    def add_tdimensions(
        self,
        node: Node,
        bounds: tuple[tuple[float, float], ...],
    ) -> None:
        """ """
        # node validations
        if not isinstance(node, Node):
            raise TypeError("node must be a Node.")
        if node not in self:
            raise ValueError("node must be contained in self.")
        # add new tbounds to the processor
        node.processor.add_tdimensions(bounds)
        return

    def remove_tdimensions(self, node: Node, dims: set[int]) -> None:
        """ """
        # node validations
        if not isinstance(node, Node):
            raise TypeError("node must be a Node.")
        if node not in self:
            raise ValueError("node must be contained in self.")
        # dims validations
        if not isinstance(dims, (set, tuple, list)):
            raise TypeError("dims must be a set.")
        if len(set(dims)) != len(dims):
            raise ValueError("All dims must be unique.")
        for dim in dims:
            if not isinstance(dim, int):
                raise TypeError("All dims must be ints.")
            if not (0 <= dim < node.tnd):
                raise IndexError("All dims must be in range.")
        # remove tdimensions from the processor
        dims = set(dims)
        node.processor.remove_tdimensions(dims)

        def _remap(old_dim: int) -> int:
            """ """
            new_dim = old_dim - sum(dim < old_dim for dim in dims)
            return new_dim

        # remove connections
        connections = []
        for connection in self._connections:
            (source, source_dim), (target, target_dim) = connection
            if source is node:
                if source_dim in dims:
                    continue
                source_dim = _remap(source_dim)
            connections.append(((source, source_dim), (target, target_dim)))
        # remove tports
        tports = []
        for port_node, port_dim in self._tports:
            if port_node is node:
                if port_dim in dims:
                    continue
                port_dim = _remap(port_dim)
            tports.append((port_node, port_dim))
        # update network
        object.__setattr__(self, "_frozen", False)
        self._connections = tuple(connections)
        self._tports = tuple(tports)
        object.__setattr__(self, "_frozen", True)
        return

    def copy(self) -> Network:
        """ """
        result = deepcopy(self)
        return result

    def to_dict(self) -> dict[str, Any]:
        """ """
        # map nodes to indexes
        node_idxs = {id(node): idx for idx, node in enumerate(self._nodes)}
        result = {
            "type": self.__class__.__name__,
            "nodes": tuple(node.to_dict() for node in self._nodes),
            "connections": tuple(
                (
                    (node_idxs[id(source)], source_dim),
                    (node_idxs[id(target)], target_dim),
                )
                for (source, source_dim), (target, target_dim) in self._connections
            ),
            "rports": tuple((node_idxs[id(node)], dim) for node, dim in self._rports),
            "tports": tuple((node_idxs[id(node)], dim) for node, dim in self._tports),
        }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Network:
        """ """
        raise NotImplementedError

    def save(self, path: str | Path, overwrite: bool = False) -> None:
        """ """
        # path validations
        if not isinstance(path, (str, Path)):
            raise TypeError("path must be a string or a Path.")
        # file validations
        path = Path(path)
        if path.suffix != ".json":
            raise ValueError("path must have a .json extension.")
        if not overwrite and path.exists():
            raise FileExistsError(f"path already exists: {path}.")
        # file creation
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=4)
        return

    @classmethod
    def load(cls, path: str | Path) -> Network:
        """ """
        # path validations
        if not isinstance(path, (str, Path)):
            raise TypeError("path must be a string or a Path.")
        # file validations
        path = Path(path)
        if path.suffix != ".json":
            raise ValueError("path must have a .json extension.")
        if not path.exists():
            raise FileNotFoundError(f"path does not exist: {path}.")
        # file loading
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        result = cls.from_dict(data)
        return result

    def _has_same_topology(self, other: Network) -> bool:
        """ """
        result = False
        if (
            len(self._nodes) == len(other._nodes)
            and len(self._connections) == len(other._connections)
            and len(self._rports) == len(other._rports)
            and len(self._tports) == len(other._tports)
        ):
            # get network topologies
            self_connections, _, _ = self._get_topology()
            other_connections, _, _ = other._get_topology()
            # get network signatures
            self_signatures = self._get_signatures()
            other_signatures = other._get_signatures()
            # find candidates for each node in self
            candidates = {
                self_idx: tuple(
                    other_idx
                    for other_idx, other_node in enumerate(other._nodes)
                    if self_node == other_node
                    and self_signatures[self_idx] == other_signatures[other_idx]
                )
                for self_idx, self_node in enumerate(self._nodes)
            }
            # ensure all nodes have candidates
            if all(len(self_candidates) > 0 for self_candidates in candidates.values()):
                mapping = {}
                used_other_idxs = set()

                def _is_compatible(self_idx: int, other_idx: int) -> bool:
                    """ """
                    result = True
                    # build candidate mappings
                    self_mapping = mapping | {self_idx: other_idx}
                    other_mapping = {
                        mapped_other_idx: mapped_self_idx
                        for mapped_self_idx, mapped_other_idx in self_mapping.items()
                    }
                    # check if connections in self are in other
                    for (self_source, self_source_dim), (
                        self_target,
                        self_target_dim,
                    ) in self_connections:
                        if self_source in self_mapping and self_target in self_mapping:
                            other_connection = (
                                (self_mapping[self_source], self_source_dim),
                                (self_mapping[self_target], self_target_dim),
                            )
                            if other_connection not in other_connections:
                                result = False
                                break
                    # check if connections in other are in self
                    for (other_source, other_source_dim), (
                        other_target,
                        other_target_dim,
                    ) in other_connections:
                        if (
                            other_source in other_mapping
                            and other_target in other_mapping
                        ):
                            self_connection = (
                                (other_mapping[other_source], other_source_dim),
                                (other_mapping[other_target], other_target_dim),
                            )
                            if self_connection not in self_connections:
                                result = False
                                break
                    return result

                def _match_nodes() -> bool:
                    """ """
                    result = False
                    # check if all nodes have been matched
                    if len(mapping) == len(self._nodes):
                        result = True
                    else:
                        # select the node with the fewest unmatched candidates
                        self_idx = min(
                            (
                                idx
                                for idx in range(len(self._nodes))
                                if idx not in mapping
                            ),
                            key=lambda idx: sum(
                                other_idx not in used_other_idxs
                                for other_idx in candidates[idx]
                            ),
                        )
                        # try to match the node with the remaining candidates
                        for other_idx in candidates[self_idx]:
                            # skip already matched candidates
                            if other_idx in used_other_idxs:
                                continue
                            # skip incompatible candidates
                            if not _is_compatible(self_idx, other_idx):
                                continue
                            # match the selected unmatched node with the candidate
                            mapping[self_idx] = other_idx
                            used_other_idxs.add(other_idx)
                            # recursively try to match the remaining unmatched nodes
                            if _match_nodes():
                                result = True
                                break
                            # backtrack if the matching fails
                            used_other_idxs.remove(other_idx)
                            del mapping[self_idx]
                    return result

                # try to match nodes with their candidates
                result = _match_nodes()
        return result

    def _get_topology(
        self,
    ) -> tuple[
        set[tuple[tuple[int, int], tuple[int, int]]],
        set[tuple[int, int]],
        set[tuple[int, int]],
    ]:
        """ """
        # map nodes to indexes
        node_idxs = {id(node): idx for idx, node in enumerate(self._nodes)}
        # get network topology
        connections = {
            (
                (node_idxs[id(source)], source_dim),
                (node_idxs[id(target)], target_dim),
            )
            for (source, source_dim), (target, target_dim) in self._connections
        }
        rports = {(node_idxs[id(node)], dim) for node, dim in self._rports}
        tports = {(node_idxs[id(node)], dim) for node, dim in self._tports}
        return connections, rports, tports

    def _get_signatures(
        self,
    ) -> tuple[
        tuple[
            tuple[tuple[int, int], ...],
            tuple[tuple[int, int], ...],
            tuple[tuple[int, int], ...],
            tuple[int, ...],
            tuple[int, ...],
        ],
        ...,
    ]:
        """ """
        # get network topology
        connections, rports, tports = self._get_topology()
        # get signatures for each node
        signatures = []
        for idx in range(len(self._nodes)):
            # get outgoing connections
            outgoing = tuple(
                sorted(
                    (source_dim, target_dim)
                    for (source, source_dim), (target, target_dim) in connections
                    if source == idx and target != idx
                )
            )
            # get incoming connections
            incoming = tuple(
                sorted(
                    (source_dim, target_dim)
                    for (source, source_dim), (target, target_dim) in connections
                    if source != idx and target == idx
                )
            )
            # get loop connections
            loops = tuple(
                sorted(
                    (source_dim, target_dim)
                    for (source, source_dim), (target, target_dim) in connections
                    if source == idx and target == idx
                )
            )
            # get rport and tport dimensions
            rdims = tuple(sorted(dim for node_idx, dim in rports if node_idx == idx))
            tdims = tuple(sorted(dim for node_idx, dim in tports if node_idx == idx))
            # create signature
            signature = (outgoing, incoming, loops, rdims, tdims)
            signatures.append(signature)
        signatures = tuple(signatures)
        return signatures

    def __setattr__(self, name: str, value: Any) -> None:
        """ """
        # freeze check
        if getattr(self, "_frozen", False):
            raise AttributeError(f"{self.__class__.__name__} is immutable.")
        # set the attribute
        object.__setattr__(self, name, value)
        return

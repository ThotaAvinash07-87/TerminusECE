"""Block diagram connection manager, DAG topological scheduler, and algebraic loop resolver."""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from .blocks import Block, ScopeSinkBlock


class SystemDiagram:
    """Manages blocks and signal connection routing."""

    def __init__(self, name: str = "System"):
        self.name = name
        self.blocks: Dict[str, Block] = {}
        # List of connections: ((src_block_name, src_port_idx), (dst_block_name, dst_port_idx))
        self.connections: List[Tuple[Tuple[str, int], Tuple[str, int]]] = []

    def clear(self) -> None:
        self.blocks.clear()
        self.connections.clear()

    def add_block(self, block: Block) -> None:
        self.blocks[block.name.upper()] = block

    def connect(self, src_endpoint: str, dst_endpoint: str) -> None:
        """Connects source output port to destination input port.
        
        Example: connect("Step1.0", "Sum1.0") or connect("Plant.out", "Scope1.in")
        """
        def parse_port(ep: str, is_src: bool) -> Tuple[str, int]:
            parts = ep.strip().split(".")
            bname = parts[0].upper()
            port = 0
            if len(parts) > 1:
                p_str = parts[1].lower()
                if p_str.isdigit():
                    port = int(p_str)
                elif p_str in ("out", "y", "output"):
                    port = 0
                elif p_str in ("in", "u", "input"):
                    port = 0
                elif p_str.startswith("in") and p_str[2:].isdigit():
                    port = int(p_str[2:])
            return bname, port

        src_b, src_p = parse_port(src_endpoint, is_src=True)
        dst_b, dst_p = parse_port(dst_endpoint, is_src=False)

        if src_b not in self.blocks:
            raise KeyError(f"Source block '{src_b}' does not exist.")
        if dst_b not in self.blocks:
            raise KeyError(f"Destination block '{dst_b}' does not exist.")

        self.connections.append(((src_b, src_p), (dst_b, dst_p)))


class Scheduler:
    """Computes valid execution order for block evaluations in dynamic systems."""

    def __init__(self, diagram: SystemDiagram):
        self.diagram = diagram
        self.execution_order: List[Block] = []
        self.stateful_blocks: List[Block] = []

    def build_schedule(self) -> List[Block]:
        """Performs topological sort on direct-feedthrough dependencies."""
        blocks = list(self.diagram.blocks.values())
        self.stateful_blocks = [b for b in blocks if b.num_states > 0]

        # In-degree of direct feedthrough dependencies
        adj: Dict[str, Set[str]] = {b.name.upper(): set() for b in blocks}
        in_degree: Dict[str, int] = {b.name.upper(): 0 for b in blocks}

        for (src_name, src_port), (dst_name, dst_port) in self.diagram.connections:
            dst_block = self.diagram.blocks[dst_name]
            if dst_block.direct_feedthrough:
                # dst depends on src
                if src_name not in adj:
                    adj[src_name] = set()
                if dst_name not in adj[src_name]:
                    adj[src_name].add(dst_name)
                    in_degree[dst_name] += 1

        # Kahn's algorithm for topological sorting
        queue = [bname for bname, deg in in_degree.items() if deg == 0]
        order_names: List[str] = []

        while queue:
            curr = queue.pop(0)
            order_names.append(curr)

            for neighbor in adj[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order_names) < len(blocks):
            # Algebraic loop detected: append remaining blocks
            for bname in self.diagram.blocks:
                if bname not in order_names:
                    order_names.append(bname)

        self.execution_order = [self.diagram.blocks[name] for name in order_names]
        return self.execution_order

    def propagate_signals(self, t: float) -> None:
        """Evaluates blocks in scheduled order and propagates output values through connections."""
        for block in self.execution_order:
            block.compute_output(t)
            # Propagate outputs to connected inputs
            for (src_name, src_port), (dst_name, dst_port) in self.diagram.connections:
                if src_name == block.name.upper():
                    dst_block = self.diagram.blocks[dst_name]
                    if src_port < len(block.outputs) and dst_port < len(dst_block.inputs):
                        dst_block.inputs[dst_port] = block.outputs[src_port]

        # Record scopes
        for block in self.execution_order:
            if isinstance(block, ScopeSinkBlock):
                block.record(t)

    def pack_states(self) -> np.ndarray:
        """Packs all block state vectors into a single continuous flat array."""
        state_list = [b.states for b in self.stateful_blocks]
        if not state_list:
            return np.zeros(0, dtype=float)
        return np.concatenate(state_list)

    def unpack_states(self, state_vec: np.ndarray) -> None:
        """Unpacks a flat state vector back into the respective block states."""
        idx = 0
        for b in self.stateful_blocks:
            n = b.num_states
            b.states = np.array(state_vec[idx : idx + n], dtype=float)
            idx += n

    def compute_all_derivatives(self, t: float) -> np.ndarray:
        """Computes concatenated time derivatives for the entire system at time t."""
        self.propagate_signals(t)
        deriv_list = [b.compute_derivatives(t) for b in self.stateful_blocks]
        if not deriv_list:
            return np.zeros(0, dtype=float)
        return np.concatenate(deriv_list)

"""Netlist manager and parser for interactive CLI circuit commands and SPICE format."""

from __future__ import annotations
import re
from typing import Dict, List, Optional, Set, Tuple, Union
from .components import (
    Component,
    Resistor,
    Capacitor,
    Inductor,
    VoltageSource,
    CurrentSource,
    Diode,
    VCVS,
    BJT,
)
from CORE.common_math import parse_eng_unit


class Netlist:
    """Manages the graph of circuit components and node connections."""

    def __init__(self, name: str = "Circuit"):
        self.name = name
        self.components: Dict[str, Component] = {}
        # Map: component_name -> list of pin net names
        self.pin_map: Dict[str, List[str]] = {}
        # Map alias nodes (e.g., 'gnd', 'GND', '0' -> '0')
        self.node_aliases: Dict[str, str] = {"gnd": "0", "GND": "0", "0": "0"}

    def clear(self) -> None:
        self.components.clear()
        self.pin_map.clear()
        self.node_aliases = {"gnd": "0", "GND": "0", "0": "0"}

    def normalize_node(self, node: str) -> str:
        n = node.strip()
        return self.node_aliases.get(n, n)

    def add_component(self, comp: Component) -> None:
        self.components[comp.name.upper()] = comp
        self.pin_map[comp.name.upper()] = [self.normalize_node(n) for n in comp.nodes]

    def remove_component(self, name: str) -> bool:
        uname = name.upper()
        if uname in self.components:
            del self.components[uname]
            if uname in self.pin_map:
                del self.pin_map[uname]
            return True
        return False

    def get_all_nodes(self) -> List[str]:
        """Returns all unique node names with '0' (ground) at index 0 if present."""
        nodes: Set[str] = set()
        for cname, pins in self.pin_map.items():
            for p in pins:
                nodes.add(self.normalize_node(p))
        
        node_list = sorted(list(nodes))
        if "0" in node_list:
            node_list.remove("0")
            node_list.insert(0, "0")
        return node_list

    def resolve_pin_index(self, comp_name: str, pin_spec: str) -> int:
        """Resolves pin name like 'p', 'n', 'a', 'b', '1', '2', 'c', 'b', 'e' to pin index."""
        comp = self.components.get(comp_name.upper())
        if not comp:
            raise KeyError(f"Component '{comp_name}' does not exist in netlist.")

        spec = pin_spec.lower().strip()

        if isinstance(comp, (Resistor, Capacitor, Inductor, VoltageSource, CurrentSource)):
            if spec in ("p", "a", "plus", "+", "1", "anode", "in"):
                return 0
            if spec in ("n", "b", "minus", "-", "2", "cathode", "out"):
                return 1
        elif isinstance(comp, Diode):
            if spec in ("a", "p", "anode", "plus", "1"):
                return 0
            if spec in ("k", "c", "n", "cathode", "minus", "2"):
                return 1
        elif isinstance(comp, BJT):
            if spec in ("c", "collector", "1"):
                return 0
            if spec in ("b", "base", "2"):
                return 1
            if spec in ("e", "emitter", "3"):
                return 2
        elif isinstance(comp, VCVS):
            if spec in ("out_p", "out+", "1"):
                return 0
            if spec in ("out_n", "out-", "2"):
                return 1
            if spec in ("in_p", "in+", "3"):
                return 2
            if spec in ("in_n", "in-", "4"):
                return 3

        # Numerical index fallback
        if spec.isdigit():
            idx = int(spec) - 1
            if 0 <= idx < len(comp.nodes):
                return idx

        raise ValueError(f"Unknown pin specification '{pin_spec}' for component '{comp_name}'.")

    def connect_pins(self, targets: List[str]) -> None:
        """Connects multiple pins and/or explicit net names together to form a common node.
        
        Example:
            targets = ["V1.p", "R1.a"]
            targets = ["R1.b", "C1.a", "node_out"]
            targets = ["V1.n", "C1.b", "gnd"]
        """
        if len(targets) < 2:
            raise ValueError("Connect command requires at least 2 endpoints/nets.")

        # Determine target net name
        chosen_net: Optional[str] = None
        pins_to_wire: List[Tuple[str, int]] = []

        for item in targets:
            item = item.strip()
            if not item:
                continue
            if "." in item:
                comp_name, pin_name = item.split(".", 1)
                comp_name_u = comp_name.upper()
                if comp_name_u not in self.components:
                    raise KeyError(f"Component '{comp_name}' not found.")
                pin_idx = self.resolve_pin_index(comp_name_u, pin_name)
                pins_to_wire.append((comp_name_u, pin_idx))
            else:
                # Explicit named net
                norm = self.normalize_node(item)
                if norm == "0":
                    chosen_net = "0"
                elif chosen_net is None:
                    chosen_net = norm

        # If no explicit net name was given, synthesize or reuse existing
        if chosen_net is None:
            # Check if any pin already has a defined node (not placeholder)
            for cname, pidx in pins_to_wire:
                curr_node = self.pin_map[cname][pidx]
                if curr_node != "?" and curr_node != "0":
                    chosen_net = curr_node
                    break
            if chosen_net is None:
                first_c, first_p = pins_to_wire[0]
                chosen_net = f"net_{first_c.lower()}_{first_p}"

        # Assign the chosen net to all pins
        for cname, pidx in pins_to_wire:
            self.pin_map[cname][pidx] = chosen_net
            self.components[cname].nodes[pidx] = chosen_net


class CircuitParser:
    """Parses interactive commands and SPICE text files into Netlist structures and simulation specs."""

    @classmethod
    def parse_command(cls, netlist: Netlist, cmd_line: str) -> Dict[str, Union[str, Dict, bool]]:
        """Parses a single interactive circuit command.
        
        Supported commands:
          add <name> <value> [options]
          connect <pin1> | <pin2> [| <pin3> ...]
          remove <name>
          set <name>.<param> <val>
          run <sim_command> (e.g. run .ac dec 10 1Hz 100kHz, run .tran 1u 10m, run .op, run .dc V1 0 10 0.1)
          list
          clear
        """
        line = cmd_line.strip()
        if not line or line.startswith("#") or line.startswith("*"):
            return {"type": "comment", "raw": line}

        parts = line.split()
        cmd = parts[0].lower()

        if cmd == "clear":
            netlist.clear()
            return {"type": "clear", "status": "ok", "message": "Circuit cleared."}

        elif cmd in ("list", "show"):
            summary = []
            for name, comp in netlist.components.items():
                pins = netlist.pin_map.get(name, comp.nodes)
                val = getattr(comp, "value", getattr(comp, "dc", ""))
                summary.append(f"{name} ({pins}) = {val}")
            return {"type": "list", "status": "ok", "components": summary}

        elif cmd == "add":
            if len(parts) < 3:
                raise ValueError("Usage: add <Name> <Value/Type> [param=val ...]")
            name = parts[1].upper()
            val_or_type = parts[2]
            kwargs = {}
            for p in parts[3:]:
                if "=" in p:
                    k, v = p.split("=", 1)
                    kwargs[k.lower()] = v

            comp = cls._create_component_from_spec(name, val_or_type, kwargs)
            netlist.add_component(comp)
            return {"type": "add", "status": "ok", "name": comp.name, "component": repr(comp)}

        elif cmd == "connect":
            raw_conns = " ".join(parts[1:])
            endpoints = [e.strip() for e in raw_conns.split("|") if e.strip()]
            if len(endpoints) < 2:
                # Try space-separated if no pipe used
                endpoints = parts[1:]
            netlist.connect_pins(endpoints)
            return {"type": "connect", "status": "ok", "endpoints": endpoints}

        elif cmd in ("remove", "delete", "del"):
            if len(parts) < 2:
                raise ValueError("Usage: remove <Name>")
            name = parts[1]
            ok = netlist.remove_component(name)
            return {"type": "remove", "status": "ok" if ok else "not_found", "name": name}

        elif cmd == "set":
            if len(parts) < 3:
                raise ValueError("Usage: set <Name>.<param> <Value>")
            target = parts[1]
            val = parts[2]
            if "." in target:
                comp_name, param = target.split(".", 1)
                comp = netlist.components.get(comp_name.upper())
                if not comp:
                    raise KeyError(f"Component '{comp_name}' not found.")
                num_val = parse_eng_unit(val)
                setattr(comp, param.lower(), num_val)
                return {"type": "set", "status": "ok", "target": target, "value": num_val}
            else:
                raise ValueError("Set target must be in format <Component>.<param>")

        elif cmd == "run":
            sim_spec_str = " ".join(parts[1:])
            return {"type": "run", "status": "ok", "sim_spec": sim_spec_str}

        raise ValueError(f"Unknown circuit command: '{cmd}'")

    @classmethod
    def _create_component_from_spec(
        cls,
        name: str,
        val_or_type: str,
        kwargs: Dict[str, str]
    ) -> Component:
        prefix = name[0].upper()

        if prefix == "R":
            return Resistor(name, "?", "?", val_or_type)
        elif prefix == "C":
            ic = float(kwargs.get("ic", 0.0))
            return Capacitor(name, "?", "?", val_or_type, ic=ic)
        elif prefix == "L":
            ic = float(kwargs.get("ic", 0.0))
            return Inductor(name, "?", "?", val_or_type, ic=ic)
        elif prefix == "V":
            ac_mag = float(parse_eng_unit(kwargs.get("ac", "0.0")))
            ac_ph = float(kwargs.get("phase", "0.0"))
            wtype = kwargs.get("type", "DC").upper()
            wave_params = {}
            if "freq" in kwargs:
                wave_params["freq"] = parse_eng_unit(kwargs["freq"])
            if "amplitude" in kwargs or "amp" in kwargs:
                wave_params["amplitude"] = parse_eng_unit(kwargs.get("amplitude", kwargs.get("amp", "1.0")))
            if "offset" in kwargs:
                wave_params["offset"] = parse_eng_unit(kwargs["offset"])
            return VoltageSource(
                name, "?", "?",
                dc=val_or_type,
                ac_mag=ac_mag,
                ac_phase=ac_ph,
                waveform_type=wtype,
                wave_params=wave_params
            )
        elif prefix == "I":
            ac_mag = float(parse_eng_unit(kwargs.get("ac", "0.0")))
            ac_ph = float(kwargs.get("phase", "0.0"))
            return CurrentSource(name, "?", "?", dc=val_or_type, ac_mag=ac_mag, ac_phase=ac_ph)
        elif prefix == "D":
            is_sat = float(kwargs.get("is", "1e-14"))
            n = float(kwargs.get("n", "1.0"))
            return Diode(name, "?", "?", is_sat=is_sat, n_ideal=n)
        elif prefix == "E":
            gain = parse_eng_unit(val_or_type)
            return VCVS(name, "?", "?", "?", "?", gain=gain)
        elif prefix == "Q":
            beta = float(kwargs.get("beta", "100.0"))
            bjt_type = kwargs.get("type", "NPN").upper()
            return BJT(name, "?", "?", "?", bjt_type=bjt_type, beta=beta)

        # Generic default as resistor
        return Resistor(name, "?", "?", val_or_type)

    @classmethod
    def parse_spice_netlist(cls, spice_text: str) -> Tuple[Netlist, List[str]]:
        """Parses a full standard SPICE netlist string."""
        netlist = Netlist()
        sim_commands: List[str] = []

        lines = spice_text.strip().splitlines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith("*"):
                continue

            if line.startswith("."):
                sim_commands.append(line)
                continue

            tokens = line.split()
            if len(tokens) < 3:
                continue

            cname = tokens[0].upper()
            prefix = cname[0]

            if prefix == "R":
                # R1 n1 n2 1k
                netlist.add_component(Resistor(cname, tokens[1], tokens[2], tokens[3]))
            elif prefix == "C":
                # C1 n1 n2 1u [IC=0]
                ic = 0.0
                if len(tokens) > 4 and tokens[4].upper().startswith("IC="):
                    ic = parse_eng_unit(tokens[4].split("=")[1])
                netlist.add_component(Capacitor(cname, tokens[1], tokens[2], tokens[3], ic=ic))
            elif prefix == "L":
                # L1 n1 n2 1m [IC=0]
                ic = 0.0
                if len(tokens) > 4 and tokens[4].upper().startswith("IC="):
                    ic = parse_eng_unit(tokens[4].split("=")[1])
                netlist.add_component(Inductor(cname, tokens[1], tokens[2], tokens[3], ic=ic))
            elif prefix == "V":
                # V1 n1 n2 [DC] 10 [AC 1 0] [SIN(offset amp freq)]
                dc_val = 0.0
                ac_mag = 0.0
                ac_ph = 0.0
                wtype = "DC"
                wparams = {}

                idx = 3
                while idx < len(tokens):
                    tok = tokens[idx].upper()
                    if tok == "DC" and idx + 1 < len(tokens):
                        dc_val = parse_eng_unit(tokens[idx + 1])
                        idx += 2
                    elif tok == "AC" and idx + 1 < len(tokens):
                        ac_mag = parse_eng_unit(tokens[idx + 1])
                        idx += 2
                        if idx < len(tokens) and not tokens[idx].startswith("."):
                            try:
                                ac_ph = float(tokens[idx])
                                idx += 1
                            except ValueError:
                                pass
                    elif tok.startswith("SIN"):
                        wtype = "SINE"
                        idx += 1
                    else:
                        try:
                            dc_val = parse_eng_unit(tok)
                        except ValueError:
                            pass
                        idx += 1

                netlist.add_component(VoltageSource(cname, tokens[1], tokens[2], dc=dc_val, ac_mag=ac_mag, ac_phase=ac_ph, waveform_type=wtype, wave_params=wparams))
            elif prefix == "I":
                # I1 n1 n2 DC 1m
                netlist.add_component(CurrentSource(cname, tokens[1], tokens[2], dc=tokens[3] if len(tokens) > 3 else "0"))
            elif prefix == "D":
                # D1 n1 n2 model
                netlist.add_component(Diode(cname, tokens[1], tokens[2]))

        return netlist, sim_commands

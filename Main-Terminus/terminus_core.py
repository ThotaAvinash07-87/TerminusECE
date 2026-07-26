from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, DataTable, Input, RichLog
from textual.containers import Horizontal
import json
import numpy as np

# =============================================================================
# PILLAR 2: THE UNIVERSAL DOM (Upgraded for Nodes)
# =============================================================================
class UniversalDOM:
    def __init__(self):
        self.components = {}
        self.nodes = set(["gnd"]) # Ground is always node 0
        self.pin_to_node = {}     # Maps pins (e.g., 'R1.a') to a node name

    def add_component(self, comp_type: str, name: str, value: str):
        self.components[name] = {"type": comp_type, "value": value}
        # Automatically generate default disconnected pins for the component
        if comp_type == 'R':
            self.pin_to_node[f"{name}.a"] = f"float_{name}_a"
            self.pin_to_node[f"{name}.b"] = f"float_{name}_b"
        elif comp_type == 'V':
            self.pin_to_node[f"{name}.p"] = f"float_{name}_p"
            self.pin_to_node[f"{name}.n"] = f"float_{name}_n"

    def connect_pin(self, pin: str, node: str):
        """Binds a component's pin to a specific circuit node."""
        self.nodes.add(node)
        self.pin_to_node[pin] = node

    def to_netlist_v1(self) -> str:
        schema = {
            "version": "1.1",
            "components": self.components,
            "netlist": self.pin_to_node
        }
        return json.dumps(schema, indent=2)

# =============================================================================
# PILLAR 3: THE FROZEN KERNEL (Real MNA Solver)
# =============================================================================
class FrozenKernel:
    def __init__(self, dom: UniversalDOM):
        self.dom = dom

    def parse_val(self, v_str: str) -> float:
        """Converts strings like '1k' or '10V' into pure floats."""
        v = v_str.lower().replace('v', '').replace('ohm', '')
        if 'k' in v: return float(v.replace('k', '')) * 1e3
        if 'm' in v: return float(v.replace('m', '')) * 1e6
        if 'u' in v: return float(v.replace('u', '')) * 1e-6
        return float(v)

    def run_dc_analysis(self) -> str:
        output = "[KERNEL] Compiling MNA Matrix...\n"

        # 1. Map string nodes to integer indices (gnd is always 0)
        node_map = {"gnd": 0}
        idx = 1
        for node in self.dom.nodes:
            if node != "gnd" and not node.startswith("float_"):
                node_map[node] = idx
                idx += 1

        num_nodes = len(node_map) - 1 # Exclude ground from the matrix size
        
        # 2. Track voltage sources (they require extra rows/cols in MNA)
        v_sources = [n for n, c in self.dom.components.items() if c['type'] == 'V']
        num_v = len(v_sources)
        total_size = num_nodes + num_v
        
        if total_size == 0:
            return "[error]Circuit is empty or missing a 'gnd' connection.[/error]"

        # Initialize Conductance (G) and Current (I) arrays
        G = np.zeros((total_size, total_size))
        I = np.zeros(total_size)

        # 3. Populate the Matrix
        try:
            for name, comp in self.dom.components.items():
                val = self.parse_val(comp["value"])
                
                if comp['type'] == 'R':
                    node_a = node_map.get(self.dom.pin_to_node[f"{name}.a"], 0)
                    node_b = node_map.get(self.dom.pin_to_node[f"{name}.b"], 0)
                    
                    g = 1.0 / val
                    if node_a > 0: G[node_a-1, node_a-1] += g
                    if node_b > 0: G[node_b-1, node_b-1] += g
                    if node_a > 0 and node_b > 0:
                        G[node_a-1, node_b-1] -= g
                        G[node_b-1, node_a-1] -= g
                        
                elif comp['type'] == 'V':
                    node_p = node_map.get(self.dom.pin_to_node[f"{name}.p"], 0)
                    node_n = node_map.get(self.dom.pin_to_node[f"{name}.n"], 0)
                    v_idx = num_nodes + v_sources.index(name)
                    
                    if node_p > 0:
                        G[node_p-1, v_idx] = 1
                        G[v_idx, node_p-1] = 1
                    if node_n > 0:
                        G[node_n-1, v_idx] = -1
                        G[v_idx, node_n-1] = -1
                        
                    I[v_idx] = val

            # 4. Solve the linear system
            solution = np.linalg.solve(G, I)
            
            # 5. Format Output
            output += "[bold green]Simulation Complete![/bold green]\n"
            for node_name, node_idx in node_map.items():
                if node_idx > 0:
                    volts = solution[node_idx-1]
                    output += f"Node [bold cyan]'{node_name}'[/bold cyan]: {volts:.4f} V\n"
                    
            for i, v_name in enumerate(v_sources):
                current = solution[num_nodes + i]
                output += f"Current through [bold cyan]{v_name}[/bold cyan]: {current*1000:.4f} mA\n"
                
            return output
            
        except np.linalg.LinAlgError:
            return "[error]Matrix is singular! Check for floating nodes or shorted voltage sources.[/error]"

# =============================================================================
# PILLAR 1: THE TUI & COMMAND PARSER 
# =============================================================================
class CommandParser:
    def __init__(self, dom: UniversalDOM):
        self.dom = dom

    def parse(self, command_line: str) -> str:
        tokens = command_line.strip().split()
        if not tokens: return ""
        action = tokens[0].lower()
        
        try:
            if action == "add":
                comp_type = tokens[1][0].upper() 
                name = tokens[1]
                value = tokens[2]
                self.dom.add_component(comp_type, name, value)
                return f"[success]Added {name} ({value})[/success]"
                
            elif action == "connect":
                # New Syntax: connect R1.a node1
                pin = tokens[1]
                node = tokens[2]
                
                if pin not in self.dom.pin_to_node:
                    return f"[error]Pin '{pin}' does not exist.[/error]"
                    
                self.dom.connect_pin(pin, node)
                return f"[success]Connected {pin} to Node '{node}'[/success]"
            
            elif action == "debug":
                return f"\n[bold yellow]Current Netlist.v1 State:[/bold yellow]\n{self.dom.to_netlist_v1()}"
                
            else:
                return f"[error]Unknown command '{action}'[/error]"
                
        except IndexError:
            return "[error]Syntax Error. Examples: 'add R1 1k' or 'connect R1.a n1'[/error]"

# =============================================================================
# THE TEXTUAL APP
# =============================================================================
class TerminusApp(App):
    TITLE = "Terminus ECE - Command Workspace"
    CSS = """
    #main_workspace { height: 1fr; }
    #console_log { width: 2fr; border: solid green; }
    #component_ledger { width: 1fr; border: solid blue; }
    #cmd_input { dock: bottom; border: tall white; }
    .success { color: green; }
    .error { color: red; }
    """

    def __init__(self):
        super().__init__()
        self.dom = UniversalDOM()
        self.parser = CommandParser(self.dom)
        self.kernel = FrozenKernel(self.dom)

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main_workspace"):
            yield RichLog(id="console_log", highlight=True, markup=True)
            yield DataTable(id="component_ledger")
        yield Input(placeholder="Type a command (e.g., 'add V1 10V', 'connect V1.p n1', 'run dc')", id="cmd_input")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(Input).focus()
        log = self.query_one(RichLog)
        log.write("[bold cyan]=== TerminusECE Workspace Initialized ===[/bold cyan]")
        
        table = self.query_one(DataTable)
        table.add_columns("Component", "Type", "Value")
        table.cursor_type = "row"
        table.zebra_stripes = True

    def on_input_submitted(self, event: Input.Submitted) -> None:
        command = event.value
        input_widget = self.query_one(Input)
        log = self.query_one(RichLog)
        
        log.write(f"> [bold]{command}[/bold]")
        
        if command.lower() == "run dc":
            result = self.kernel.run_dc_analysis()
            log.write(result)
        elif command.lower() == "clear":
            log.clear()
        elif command.strip():
            result = self.parser.parse(command)
            log.write(result)
            self.update_ledger()
            
        input_widget.value = ""

    def update_ledger(self) -> None:
        table = self.query_one(DataTable)
        table.clear()
        for name, data in self.dom.components.items():
            table.add_row(name, data["type"], data["value"])

if __name__ == "__main__":
    TerminusApp().run()
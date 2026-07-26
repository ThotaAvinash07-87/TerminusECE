from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, DataTable, Input, RichLog
from textual.containers import Horizontal
import json
import numpy as np

# =============================================================================
# PILLAR 2: THE UNIVERSAL DOM
# =============================================================================
class UniversalDOM:
    def __init__(self):
        self.components = {}
        self.connections = []
    
    def add_component(self, comp_type: str, name: str, value: str):
        self.components[name] = {"type": comp_type, "value": value, "pins": []}
        
    def add_connection(self, source_pin: str, target_pin: str):
        self.connections.append((source_pin, target_pin))

    def to_netlist_v1(self) -> str:
        schema = {
            "version": "1.0",
            "components": self.components,
            "connections": self.connections
        }
        return json.dumps(schema, indent=2)

# =============================================================================
# PILLAR 3: THE FROZEN KERNEL
# =============================================================================
class FrozenKernel:
    def __init__(self, dom: UniversalDOM):
        self.dom = dom

    def run_dc_analysis(self) -> str:
        # Returns a string so the UI can display it in the log
        output = "[KERNEL] Locking DOM state...\n"
        
        # Dummy matrix math for demonstration
        output += "[KERNEL] Solving G * v = I using numpy.linalg.solve...\n"
        G = np.array([[ 0.001, -0.001], [-0.001,  0.001]])
        I = np.array([0.01, 0])
        
        v = [10.0, 9.99] # Mock result
        output += f"[KERNEL] Simulation Complete. Node Voltages: {v}\n"
        return output

# =============================================================================
# PILLAR 1: THE TUI & COMMAND PARSER (Textual Integration)
# =============================================================================
class CommandParser:
    def __init__(self, dom: UniversalDOM):
        self.dom = dom

    def parse(self, command_line: str) -> str:
        """Parses the command and returns a message for the UI Log."""
        tokens = command_line.strip().split()
        if not tokens:
            return ""
            
        action = tokens[0].lower()
        
        try:
            if action == "add":
                comp_type = tokens[1][0].upper() 
                name = tokens[1]
                value = tokens[2]
                self.dom.add_component(comp_type, name, value)
                return f"[success]Added {name} ({value}) to the DOM.[/success]"
                
            elif action == "connect":
                pins = [t for t in tokens[1:] if t != "|"]
                for i in range(len(pins)-1):
                    self.dom.add_connection(pins[i], pins[i+1])
                return f"[success]Connected {' -> '.join(pins)}[/success]"
            
            elif action == "debug":
                return f"\n[bold yellow]Current Netlist.v1 State:[/bold yellow]\n{self.dom.to_netlist_v1()}"
                
            else:
                return f"[error]Error: Unknown command '{action}'[/error]"
                
        except IndexError:
            return "[error]Error: Invalid command syntax. Example: 'add R1 1k'[/error]"


# =============================================================================
# THE TEXTUAL APP
# =============================================================================
class TerminusApp(App):
    TITLE = "Terminus ECE - Command Workspace"
    
    # CSS to position the input at the bottom and split the screen
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
            # The RichLog acts as the terminal output area
            yield RichLog(id="console_log", highlight=True, markup=True)
            # The DataTable acts as the live component ledger
            yield DataTable(id="component_ledger")
        # The Input acts as the command line
        yield Input(placeholder="Type a command (e.g., 'add V1 10V', 'connect V1.p | R1.a', 'run dc')", id="cmd_input")
        yield Footer()

    def on_mount(self) -> None:
        """Setup the tables and log when the app starts."""
        self.query_one(Input).focus() # Put cursor in the input box immediately
        
        log = self.query_one(RichLog)
        log.write("[bold cyan]=== TerminusECE Workspace Initialized ===[/bold cyan]")
        log.write("Try typing: [bold]add R1 1k[/bold] or [bold]debug[/bold]")

        table = self.query_one(DataTable)
        table.add_columns("Component", "Type", "Value")
        table.cursor_type = "row"
        table.zebra_stripes = True

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """This runs every time the user presses Enter in the Input box."""
        command = event.value
        input_widget = self.query_one(Input)
        log = self.query_one(RichLog)
        
        # 1. Echo the user's command to the log
        log.write(f"> [bold]{command}[/bold]")
        
        # 2. Process the command
        if command.lower() == "run dc":
            result = self.kernel.run_dc_analysis()
            log.write(result)
        elif command.lower() == "clear":
            log.clear()
        elif command.strip():
            # Pass everything else to the parser
            result = self.parser.parse(command)
            log.write(result)
            
            # Refresh the live ledger
            self.update_ledger()
            
        # 3. Clear the input box for the next command
        input_widget.value = ""

    def update_ledger(self) -> None:
        """Clears and repopulates the DataTable with the live DOM state."""
        table = self.query_one(DataTable)
        table.clear()
        for name, data in self.dom.components.items():
            table.add_row(name, data["type"], data["value"])


if __name__ == "__main__":
    TerminusApp().run()
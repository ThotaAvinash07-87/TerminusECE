from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, RichLog

class TerminusApp(App):
    """A Textual TUI for the Terminus ECE Workbench."""
    
    # CSS styling for basic layout
    CSS = """
    #command_input {
        dock: bottom;
        margin: 1;
    }
    #console {
        height: 100%;
        margin: 1 2;
        border: solid green;
    }
    """

    # Global keyboard shortcuts
    BINDINGS = [
        ("d", "toggle_dark", "Toggle dark mode"), 
        ("q", "quit_app", "Quit Terminus")
    ]

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header(show_clock=True)
        yield RichLog(id="console", highlight=True, markup=True)
        yield Input(placeholder="Enter command (e.g., 'help', 'mode ltspice')...", id="command_input")
        yield Footer()

    def on_ready(self) -> None:
        """Called when the DOM is ready."""
        log = self.query_one(RichLog)
        log.write("[bold green]=== Terminus Workbench Initialized ===[/bold green]")
        log.write("System ready. Awaiting commands...")
        self.query_one(Input).focus()

    def action_quit_app(self) -> None:
        """Exit the application."""
        self.exit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle when the user presses Enter in the input box."""
        log = self.query_one(RichLog)
        command = event.value.strip()
        
        if not command:
            return
        
        log.write(f"[bold cyan]> {command}[/bold cyan]")
        
        self._route_command(command, log)
        
        event.input.value = ""

    def _route_command(self, command: str, log: RichLog) -> None:
        """Validates and routes commands to the appropriate subsystem."""
        cmd_lower = command.lower()
        
        valid_modes = ["matlab", "ltspice", "simulink", "xilinx", "embedded"]
        
        if cmd_lower == "help":
            log.write(f"Available subsystems: [yellow]{', '.join(valid_modes)}[/yellow]")
            log.write("Type 'mode <subsystem>' to switch contexts.")
            
        elif cmd_lower.startswith("mode "):
            requested_mode = cmd_lower.replace("mode ", "").strip()
            
            if requested_mode in valid_modes:
                log.write(f"Switching context to: [bold magenta]{requested_mode.upper()}[/bold magenta]")
            else:
                log.write(f"[red]Context Error:[/red] Subsystem '{requested_mode}' not found.")
                log.write(f"Valid options are: {', '.join(valid_modes)}")
                
        else:
            log.write(f"[red]Syntax Error:[/red] Unknown command '{command}'")

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    app = TerminusApp()
    app.run()
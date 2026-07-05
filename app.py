from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, DataTable

class MyApp(App):
    # We can add a title that shows up in the Header
    TITLE = "Terminus ECE - Hardware Monitor"

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable()
        yield Footer()

    # on_mount runs exactly once when the application first starts up
    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        
        # Add columns to our table
        table.add_columns("Rotor ID", "Target RPM", "Current RPM", "Status")
        
        # Add rows of data
        table.add_rows([
            ("M1-FrontLeft", "8500", "8492", "STABLE"),
            ("M2-FrontRight", "8500", "8510", "STABLE"),
            ("M3-BackLeft", "8500", "8200", "WARNING"),
            ("M4-BackRight", "8500", "8499", "STABLE"),
        ])
        
        # This makes the table visually span the whole screen
        table.cursor_type = "row"
        table.zebra_stripes = True
if __name__ == "__main__":
    MyApp().run()        
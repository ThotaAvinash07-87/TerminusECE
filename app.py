from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Label

class MyApp(App):
    TITLE = "TerminusECE"
    # The compose function tells the app what to draw on the screen
    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Hello, World! I am building a terminal app.")
        yield Footer()

if __name__ == "__main__":
    MyApp().run()
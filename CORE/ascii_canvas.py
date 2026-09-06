"""2D ASCII/Unicode Canvas rendering, A* wire routing, and terminal waveform/Bode plotting."""

from __future__ import annotations
import math
import heapq
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np

BOX_CHARS = {
    "h": "─",
    "v": "│",
    "tl": "┌",
    "tr": "┐",
    "bl": "└",
    "br": "┘",
    "cross": "┼",
    "t_down": "┬",
    "t_up": "┴",
    "t_right": "├",
    "t_left": "┤",
    "arrow_r": "►",
    "arrow_l": "◄",
    "arrow_u": "▲",
    "arrow_d": "▼",
    "dot": "•",
    "circle": "○",
    "block": "█",
    "sub_block": "▀",
}


class RoutePath:
    """Represents a routed connection between 2D points on a canvas."""
    def __init__(self, points: List[Tuple[int, int]], label: str = ""):
        self.points = points
        self.label = label


class AsciiCanvas:
    """2D character grid with drawing primitives, wire routing, and component blocks."""
    
    def __init__(self, width: int = 80, height: int = 24, bg_char: str = " "):
        self.width = width
        self.height = height
        self.bg_char = bg_char
        self.grid = [[bg_char for _ in range(width)] for _ in range(height)]
        self.obstacles: Set[Tuple[int, int]] = set()
        self.junctions: Set[Tuple[int, int]] = set()

    def clear(self) -> None:
        self.grid = [[self.bg_char for _ in range(self.width)] for _ in range(self.height)]
        self.obstacles.clear()
        self.junctions.clear()

    def set_char(self, x: int, y: int, char: str) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = char[0] if char else " "

    def get_char(self, x: int, y: int) -> str:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        return " "

    def draw_text(self, x: int, y: int, text: str) -> None:
        """Writes text horizontally starting at (x, y)."""
        if not (0 <= y < self.height):
            return
        for i, ch in enumerate(text):
            px = x + i
            if 0 <= px < self.width:
                self.grid[y][px] = ch

    def draw_box(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        title: str = "",
        as_obstacle: bool = True
    ) -> None:
        """Draws a rectangular box with corners and title."""
        if w < 2 or h < 2:
            return
        
        # Corners
        self.set_char(x, y, BOX_CHARS["tl"])
        self.set_char(x + w - 1, y, BOX_CHARS["tr"])
        self.set_char(x, y + h - 1, BOX_CHARS["bl"])
        self.set_char(x + w - 1, y + h - 1, BOX_CHARS["br"])

        # Horizontal borders
        for ix in range(x + 1, x + w - 1):
            self.set_char(ix, y, BOX_CHARS["h"])
            self.set_char(ix, y + h - 1, BOX_CHARS["h"])

        # Vertical borders
        for iy in range(y + 1, y + h - 1):
            self.set_char(x, iy, BOX_CHARS["v"])
            self.set_char(x + w - 1, iy, BOX_CHARS["v"])

        # Optional Title
        if title:
            max_len = w - 4
            disp_title = f" {title[:max_len]} " if max_len > 0 else title[:w-2]
            self.draw_text(x + 2, y, disp_title)

        if as_obstacle:
            for iy in range(y, y + h):
                for ix in range(x, x + w):
                    self.obstacles.add((ix, iy))

    def route_wire(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        arrow_end: bool = False
    ) -> List[Tuple[int, int]]:
        """A* / Rectilinear router finding collision-free orthogonal path between points."""
        start_x, start_y = start
        end_x, end_y = end

        # Quick Manhattan A* pathfinding
        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set: List[Tuple[int, int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0, 0, start))
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        cost_so_far: Dict[Tuple[int, int], int] = {start: 0}

        found = False
        while open_set:
            _, _, current = heapq.heappop(open_set)
            if current == end:
                found = True
                break

            cx, cy = current
            # Orthogonal 4-connectivity
            neighbors = [(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)]
            for nx, ny in neighbors:
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                # Obstacle penalty (unless start or end)
                if (nx, ny) in self.obstacles and (nx, ny) != end and (nx, ny) != start:
                    continue

                # Turn penalty to prefer straight wires
                prev = came_from[current]
                turn_penalty = 0
                if prev is not None:
                    dx1, dy1 = cx - prev[0], cy - prev[1]
                    dx2, dy2 = nx - cx, ny - cy
                    if (dx1, dy1) != (dx2, dy2):
                        turn_penalty = 2

                new_cost = cost_so_far[current] + 1 + turn_penalty
                neighbor = (nx, ny)
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, end)
                    heapq.heappush(open_set, (priority, new_cost, neighbor))
                    came_from[neighbor] = current

        # Reconstruct path
        path: List[Tuple[int, int]] = []
        if found:
            curr: Optional[Tuple[int, int]] = end
            while curr is not None:
                path.append(curr)
                curr = came_from.get(curr)
            path.reverse()
        else:
            # Fallback direct L-bend
            path = []
            mid_x = (start_x + end_x) // 2
            step_x = 1 if end_x >= start_x else -1
            step_y = 1 if end_y >= start_y else -1
            
            for ix in range(start_x, mid_x + step_x, step_x):
                path.append((ix, start_y))
            for iy in range(start_y, end_y + step_y, step_y):
                path.append((mid_x, iy))
            for ix in range(mid_x, end_x + step_x, step_x):
                path.append((ix, end_y))

        # Render wire characters onto grid
        for i in range(len(path)):
            px, py = path[i]
            prev_p = path[i - 1] if i > 0 else None
            next_p = path[i + 1] if i < len(path) - 1 else None

            # Determine wire character
            if prev_p is None and next_p is None:
                char = BOX_CHARS["dot"]
            elif prev_p is None: # Start
                if next_p[0] != px:
                    char = BOX_CHARS["h"]
                else:
                    char = BOX_CHARS["v"]
            elif next_p is None: # End
                if arrow_end:
                    if prev_p[0] < px:
                        char = BOX_CHARS["arrow_r"]
                    elif prev_p[0] > px:
                        char = BOX_CHARS["arrow_l"]
                    elif prev_p[1] < py:
                        char = BOX_CHARS["arrow_d"]
                    else:
                        char = BOX_CHARS["arrow_u"]
                else:
                    if prev_p[0] != px:
                        char = BOX_CHARS["h"]
                    else:
                        char = BOX_CHARS["v"]
            else: # Intermediate corner / line
                dx1, dy1 = prev_p[0] - px, prev_p[1] - py
                dx2, dy2 = next_p[0] - px, next_p[1] - py

                if dx1 != 0 and dx2 != 0: # Horizontal straight
                    char = BOX_CHARS["h"]
                elif dy1 != 0 and dy2 != 0: # Vertical straight
                    char = BOX_CHARS["v"]
                else: # Corner
                    dirs = {(dx1, dy1), (dx2, dy2)}
                    if {(0, -1), (1, 0)} == dirs:
                        char = BOX_CHARS["bl"]
                    elif {(0, -1), (-1, 0)} == dirs:
                        char = BOX_CHARS["br"]
                    elif {(0, 1), (1, 0)} == dirs:
                        char = BOX_CHARS["tl"]
                    elif {(0, 1), (-1, 0)} == dirs:
                        char = BOX_CHARS["tr"]
                    else:
                        char = BOX_CHARS["cross"]

            # Check if crossing existing wire
            current_ch = self.get_char(px, py)
            if current_ch in (BOX_CHARS["h"], BOX_CHARS["v"], BOX_CHARS["cross"]) and current_ch != char:
                char = BOX_CHARS["cross"]

            self.set_char(px, py, char)

        return path

    def render(self) -> str:
        """Returns the entire canvas rendered as a multi-line string."""
        return "\n".join("".join(row) for row in self.grid)


class AsciiPlotter:
    """High-contrast ASCII / Unicode curve and multi-series plotter."""
    
    BLOCKS = [" ", " ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
    BRAILLE_DOTS = ["·", "•", "■", "♦", "*", "o", "+", "x"]

    @classmethod
    def plot(
        cls,
        x: Sequence[float],
        y: Sequence[float],
        width: int = 70,
        height: int = 15,
        title: str = "",
        x_label: str = "Time (s)",
        y_label: str = "Voltage (V)",
        marker: str = "•"
    ) -> str:
        """Generates a formatted 2D ASCII waveform plot."""
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        if len(x_arr) == 0 or len(y_arr) == 0:
            return "[Empty Plot Data]"

        x_min, x_max = float(np.min(x_arr)), float(np.max(x_arr))
        y_min, y_max = float(np.min(y_arr)), float(np.max(y_arr))

        if x_max == x_min:
            x_max += 1.0
        if y_max == y_min:
            y_max += 1.0
            y_min -= 1.0

        # Create character matrix
        plot_w = max(20, width - 15)
        plot_h = max(6, height - 4)

        grid = [[" " for _ in range(plot_w)] for _ in range(plot_h)]

        # Interpolate points onto screen
        interp_x = np.linspace(x_min, x_max, plot_w * 3)
        interp_y = np.interp(interp_x, x_arr, y_arr)

        for px_val, py_val in zip(interp_x, interp_y):
            col = int((px_val - x_min) / (x_max - x_min) * (plot_w - 1))
            row = int((py_val - y_min) / (y_max - y_min) * (plot_h - 1))
            row = plot_h - 1 - row  # invert y (0 at top)
            if 0 <= col < plot_w and 0 <= row < plot_h:
                grid[row][col] = marker

        # Build final display string with Y-axis numbers
        lines = []
        if title:
            lines.append(f"  [bold cyan]{title}[/bold cyan]".center(width))

        for r in range(plot_h):
            y_val = y_max - (r / (plot_h - 1)) * (y_max - y_min)
            row_str = "".join(grid[r])
            if r == 0:
                y_label_str = f"{y_val:8.2e} ┌"
            elif r == plot_h - 1:
                y_label_str = f"{y_val:8.2e} └"
            elif r == plot_h // 2:
                y_label_str = f"{y_val:8.2e} ┤"
            else:
                y_label_str = f"{y_val:8.2e} │"
            lines.append(f"{y_label_str}{row_str}")

        # X-axis line and ticks
        x_axis_line = " " * 10 + "└" + "─" * (plot_w - 1)
        lines.append(x_axis_line)
        x_ticks = f"{' ' * 10}{x_min:8.2e}{' ' * (plot_w - 20)}{x_max:8.2e}"
        lines.append(x_ticks)
        if x_label:
            lines.append(f"{' ' * (10 + plot_w // 2 - len(x_label) // 2)}{x_label}")

        return "\n".join(lines)


class AsciiBodePlotter:
    """Renders Frequency Response (Magnitude in dB and Phase in degrees) in ASCII."""
    
    @classmethod
    def plot_bode(
        cls,
        frequencies: Sequence[float],
        magnitude_db: Sequence[float],
        phase_deg: Sequence[float],
        width: int = 72,
        height_each: int = 10,
        title: str = "Bode Plot"
    ) -> str:
        """Generates combined magnitude and phase frequency response plots."""
        freq_arr = np.asarray(frequencies, dtype=float)
        mag_arr = np.asarray(magnitude_db, dtype=float)
        ph_arr = np.asarray(phase_deg, dtype=float)

        if len(freq_arr) == 0:
            return "[Empty Frequency Data]"

        log_freq = np.log10(np.where(freq_arr <= 0, 1e-12, freq_arr))

        mag_plot = AsciiPlotter.plot(
            log_freq,
            mag_arr,
            width=width,
            height=height_each,
            title=f"{title} - Magnitude",
            x_label="log10(Frequency [Hz])",
            y_label="Gain (dB)",
            marker="━"
        )

        phase_plot = AsciiPlotter.plot(
            log_freq,
            ph_arr,
            width=width,
            height=height_each,
            title=f"{title} - Phase",
            x_label="log10(Frequency [Hz])",
            y_label="Phase (°)",
            marker="╍"
        )

        return f"{mag_plot}\n\n{phase_plot}"

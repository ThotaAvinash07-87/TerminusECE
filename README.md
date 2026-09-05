# TerminusECE

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![Status: Concept/RFC](https://img.shields.io/badge/Status-RFC-blue.svg)]()

> The command-driven, terminal-based unified workspace for Electrical and Computer Engineering.

The ECE software ecosystem is broken by fragmentation and bloat. Students and professionals juggle heavy, proprietary GUI applications—MATLAB for signals, LTspice for circuits, Simulink for systems. 

TerminusECE replaces this fragmentation with a single, lightning-fast Terminal User Interface (TUI). We operate on a core philosophy: Engineering is math, and math is best expressed as text.

---

## Why TerminusECE?

* **The Speed of Thought:** Stop dragging wires and clicking nested menus. Power users operate faster on a keyboard. Define a control loop or a circuit in seconds using intuitive terminal syntax.
* **Featherweight Execution:** GUIs with heavy 3D rendering engines consume massive RAM. TerminusECE is a pure TUI. It runs instantly on older hardware without spinning up your fans.
* **Complete Unification:** Never leave the terminal. Signal processing, transform calculus, and transistor-level simulations all happen in the exact same unified workspace.
* **Inter-Process Sync:** Open multiple terminal windows for different subsystems. A background daemon allows them to share data instantly, meaning you can pipe a simulated transient waveform directly into a matrix solver without exporting files.
* **Hybrid Cloud Ready:** Run your daily engineering tasks locally. If you need a massive Monte Carlo yield analysis, just type `set solver=cloud` to seamlessly offload the heavy matrix math to a remote server.

---

## The Workflow (15-Minute MVP)

Design, route, and simulate directly from the command bar. Here is what an RC Low-Pass Filter simulation looks like in TerminusECE:

```sh
# 1. Define Components
terminus> add V1 10V ac=1
terminus> add R1 1k
terminus> add C1 1u

# 2. Route the Netlist
terminus> connect V1.p | R1.a
terminus> connect R1.b | C1.a | node_out
terminus> connect V1.n | C1.b | gnd

# 3. Simulate and Visualize
terminus> run .ac dec 10 1Hz 100kHz
[Terminal instantly renders a high-contrast ASCII Bode plot]

# 4. Export
terminus> export report --format=csv
```

---

## Core Architecture

To become the Linux of EDA, TerminusECE relies on a strict, decoupled architecture. The visual interface never directly touches the underlying mathematics.

* **UI (/ui):** Built with the Textual Python framework. This manages the visual workspace, screens, and ASCII rendering. It translates human-readable commands into strict internal calls.
* **Core (/core):** The backbone of the application. It contains the Inter-Process Communication (IPC) daemon for cross-terminal data sharing, ASCII 2D canvas routing, and shared math utilities.
* **The Engines (/engines):** The computational backends. These are isolated modules executing standardized mathematics:
    * **Circuit:** A Modified Nodal Analysis (MNA) solver for DC, AC, and transient circuit simulations.
    * **Numerical:** Handles matrix mathematics, custom AST parsers, and dedicated solvers for Fourier, Laplace, and Z-transforms to analyze bounded signals.
    * **Dynamic Systems:** A block-diagram topology scheduler and RK4 ODE integrator designed for control systems.
    * **Digital Logic:** Evaluates combinational and sequential logic, handling gate delays and clock edges.
    * **Embedded:** A terminal text editor paired with a lightweight register and memory emulator (supporting architectures like the TI C2000 F28069) to step through instructions and simulate peripheral outputs.

---

## The Century-Scale Covenant

This project is built to survive for decades. We commit to the following principles:

* **Frozen Math:** A netlist simulating a transistor today must give identical results fifty years from now.
* **Plugin-First:** The vast majority of future functionality, like quantum device modeling or advanced antenna solvers, will live in sandboxed plugins. The core will remain lightweight and untouched.
* **Education First:** The platform has to pass the lab test. A second-year undergraduate must be able to complete standard Signals and Systems or Circuits labs using only this tool.

---

## Roadmap (Phase 1)

We are currently building the Minimum Viable Product (MVP) core shell and the foundational mathematics.

* [x] Define the multi-engine directory architecture and IPC strategy.
* [x] Initialize the Textual TUI entrypoint and command router (terminus.py).
* [ ] Circuit Engine: Build the internal components.py and the core MNA matrix solver.
* [ ] Core Framework: Implement the ASCII canvas router to visualize component blocks in the terminal.
* [ ] Numerical Engine: Develop the NumPy bridge to evaluate strings of matrix operations.
* [ ] Release v0.1 for community testing.

---

## Contributing

TerminusECE is built by engineers, for engineers. Whether you are hacking on Python, writing DSP algorithms, or just tired of paying for heavy software licenses, we would love your help.

Read our CONTRIBUTING.md to get started. Join the effort and let's engineer at the speed of thought.

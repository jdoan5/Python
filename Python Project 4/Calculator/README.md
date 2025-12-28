# Calculator (Tkinter) — Simple + Scientific

Two desktop calculators built with **Python** and **Tkinter**:

- **Simple Calculator** — standard four operations + clear/backspace + keyboard support  
- **Scientific Calculator** — extended functions (trig/log/powers), safer expression handling, and richer keyboard shortcuts

No third‑party dependencies are required to run either app from source.

---

## Quick start

### 1) Verify Python + Tkinter

- **Windows / macOS:** Tkinter is typically included with the official Python installer.
- **Linux (Ubuntu/Debian):** you may need:

```bash
sudo apt-get install python3-tk
```

(Optional) quick GUI check:

```bash
python tk_test.py
```

---

### 2) Run the apps

> These filenames include spaces — wrap them in quotes.

**macOS / Linux**
```bash
python3 "Simple Calculator.py"
python3 "Scientific Calculator.py"
```

**Windows (PowerShell)**
```powershell
py "Simple Calculator.py"
py "Scientific Calculator.py"
```

---

## What’s included

### Simple Calculator
- Basic operations: `+`, `−`, `×`, `÷`, decimals (`.`)
- Utility keys: **C** (clear all), **CE** (clear entry), **⌫** (backspace)
- Unary keys: **x²** (square), **1/x** (reciprocal), **+/−** (negate)
- Percent key with calculator-style behavior (example: `200 + 10% → 220`)
- Keyboard support: `0–9 . + - * / ( )` and Enter/Esc/Backspace

### Scientific Calculator
Includes the Simple Calculator behavior plus:
- Scientific functions (e.g., trig, inverse trig, logarithms, powers)
- Improved keyboard handling and clear-entry behavior
- Safer evaluation via allow-listed input patterns (to avoid arbitrary code execution)

---

## Files

- `Simple Calculator.py` — main Simple Calculator app  
- `Scientific Calculator.py` — main Scientific Calculator app  
- `tk_test.py` — optional Tkinter smoke test (opens a small window)  
- `README.md` — this file

> Tip: `venv/` (or `.venv/`) should usually be excluded from Git via `.gitignore`.

---

## Troubleshooting

### Tkinter import errors
- On Linux, install `python3-tk` (see steps above).
- On Windows/macOS, reinstall Python from python.org and ensure “tcl/tk” is included.

### macOS button colors
Some macOS themes may ignore Tk button background colors; text and layout remain consistent.

---

## Portfolio demo pages (optional)

If you’re viewing this project through your portfolio site, you can link to your demo pages:
- `demo1.html` — Simple Calculator walkthrough video
- `demo2.html` — Scientific Calculator walkthrough video

# Python Tkinter Calculator

A simple desktop calculator built with **Python** and **Tkinter**.  
Run it directly with Python or package it into native apps for macOS/Windows.

---

## Features

- Clean, minimal UI using Tkinter
- Basic operations: `+`, `−`, `×`, `÷`, decimals (`.`)
- Utility keys: **C** (clear all), **CE** (clear entry), **⌫** (backspace)
- Unary keys: **x²** (square), **1/x** (reciprocal), **+/−** (negate)
- **%** with calculator-style behavior (e.g., `200 + 10% → 220`)
- Typing supported: `0–9 . + - * / ( )` and Enter/Esc/Backspace
- Cross-platform: Windows, macOS, Linux (GUI required)

> Note: On macOS, some themes ignore Tk button background colors; text color still applies.

---

## Files

- `calculator.py` — main Tkinter calculator app  
- `tk_test.py` *(optional)* — tiny script to confirm Tkinter can open a window

No third-party pip packages are required to run from source.

---

## Requirements

- **Python 3.9+** (3.11+ recommended)
- **Tkinter**
  - Included by default with the official installers on **Windows** and **macOS**.
  - On some Linux distros: `sudo apt-get install python3-tk`

---

## How to Run (from source)

Open a terminal (or PyCharm’s terminal) and run:

### macOS / Linux
```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python calculator.py
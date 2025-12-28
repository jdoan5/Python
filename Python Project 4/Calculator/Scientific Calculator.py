# Extended version of a Tkinter calculator with more scientific calculation functionality

import tkinter as tk
import math
import random
import traceback


class ScientificCalculator(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Scientific Calculator")
        self.configure(bg="#222222")

        self.expression = tk.StringVar()

        # --- Display ---------------------------------------------------------
        display = tk.Entry(
            self,
            textvariable=self.expression,
            font=("Segoe UI", 18),  # <- bigger font makes it taller
            bd=0,
            relief="flat",
            justify="right",
            bg="#0d3b66",
            fg="white",
            insertbackground="white",
        )
        display.grid(
            row=0, column=0, columnspan=5,
            sticky="nsew",
            padx=4, pady=4,
            ipady=16,  # <- this is what really “expands” the height
        )

        # --- Button layout (labels only) ------------------------------------
        buttons = [
            ["(", ")", "C", "CE", "⌫"],
            ["%", "1/x", "√", "π", "Rand"],
            ["sin", "cos", "tan", "x²", "x³"],
            ["sin⁻¹", "cos⁻¹", "tan⁻¹", "x!", "÷"],
            ["e", "7", "8", "9", "×"],
            ["ln", "4", "5", "6", "−"],
            ["log₁₀", "1", "2", "3", "+"],
            ["+/-", ".", "0", "Ans", "="],
        ]

        for r, row in enumerate(buttons, start=1):
            for c, label in enumerate(row):
                bg = self._bg_for(label)
                fg = "white" if label in {"C", "CE", "=", "⌫"} else "black"

                w = tk.Label(
                    self,
                    text=label,
                    font=("Segoe UI", 20),
                    bg=bg,
                    fg=fg,
                    bd=1,
                    relief="flat",
                    padx=6,
                    pady=12,
                )
                w.grid(row=r, column=c, sticky="nsew", padx=1, pady=1)

                # make the label behave like a button
                w.bind("<Button-1>", lambda e, v=label: self.on_button(v))
                w.bind("<Enter>", lambda e, widget=w: widget.config(relief="raised"))
                w.bind("<Leave>", lambda e, widget=w: widget.config(relief="flat"))


        # make grid responsive
        for col in range(5):
            self.grid_columnconfigure(col, weight=1)
        for row in range(1, 9):
            self.grid_rowconfigure(row, weight=1)

        # store last answer for "Ans"
        self.last_answer = None

    # --------------------------------------------------------------------- #
    # styling helper
    def _bg_for(self, label: str) -> str:
        if label == "C":
            return "#e74c3c"  # red
        if label == "CE":
            return "#ff9800"  # orange
        if label == "⌫":
            return "#6d4c41"  # brown
        if label == "=":
            return "#4caf50"  # green
        if label in {"÷", "×", "−", "+"}:
            return "#ADD8E6"  # light blue
        return "#f2f0eb"  # gray for others

    # --------------------------------------------------------------------- #
    # button behaviour
    def on_button(self, label: str):
        if label == "C":          # clear everything
            self.expression.set("")
            self.last_answer = None
            return

        if label == "CE":         # clear entry
            self.expression.set("")
            return

        if label == "⌫":          # backspace
            self.expression.set(self.expression.get()[:-1])
            return

        if label == "=":
            self._calculate()
            return

        if label == "+/-":
            expr = self.expression.get()
            if expr.startswith("-"):
                self.expression.set(expr[1:])
            else:
                self.expression.set("-" + expr if expr else "-")
            return

        if label == "Ans" and self.last_answer is not None:
            self.expression.set(self.expression.get() + str(self.last_answer))
            return

        if label == "%":
            expr = self.expression.get()
            if not expr:
                return
            try:
                value = float(expr)
                result = value / 100
                self.last_answer = result
                self.expression.set(str(result))
            except ValueError:
                # If it is not a simple number, just append "/100"
                self.expression.set(expr + "/100")
                return

        # Otherwise append translated token to the expression string
        self.expression.set(self.expression.get() + self._to_expr_token(label))

    # map button labels to expression tokens used by eval()
    def _to_expr_token(self, label: str) -> str:
        mapping = {
            "×": "*",
            "÷": "/",
            "−": "-",
            "π": "pi",
            "e": "e",
            "√": "sqrt(",
            "1/x": "1/(",
            "sin": "sin(",
            "cos": "cos(",
            "tan": "tan(",
            "sin⁻¹": "asin(",
            "cos⁻¹": "acos(",
            "tan⁻¹": "atan(",
            "ln": "ln(",
            "log₁₀": "log10(",
            "x²": "**2",
            "x³": "**3",
            "x!": "fact(",
            "%": "%",
            # simple “Rand” inserts a random number 0–1
            "Rand": "rand()",
        }

        if label in mapping:
            return mapping[label]
        # digits, parentheses, dot, %, +, etc. are used as-is
        return label

    def _calculate(self):
        expr = self.expression.get()

        # safe evaluation environment
        env = {
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "asin": math.asin,
            "acos": math.acos,
            "atan": math.atan,
            "sqrt": math.sqrt,
            "pi": math.pi,
            "e": math.e,
            "ln": math.log,         # natural log
            "log10": math.log10,
            "fact": math.factorial,
            "rand": random.random,
        }

        try:
            # NOTE: trig functions use radians (standard for math module)
            result = eval(expr, {"__builtins__": {}}, env)
            self.last_answer = result
            self.expression.set(str(result))
        except Exception:
            self.expression.set("Error")


if __name__ == "__main__":
    app = ScientificCalculator()
    app.mainloop()
import tkinter as tk
from tkinter import ttk, messagebox
import re

# Very small safety gate for eval: digits, dot, + - * / ( ) and whitespace only
MATH_ALLOWED = re.compile(r"^[0-9\.\+\-\*\/\(\)\s]+$")

class Calculator(tk.Tk):
    SYMBOL_MAP = {"÷": "/", "×": "*", "−": "-"}

    def __init__(self):
        super().__init__()
        self.title("Python Calculator")
        self.resizable(True, True)
        self.expression = ""

        # ---- TTK theme & styles (works on macOS) ---------------------------
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "Calc.TButton",
            font=("Segoe UI", 18),
            foreground="#111111",
            background="#e9e9e9",
            padding=(8, 12),
            borderwidth=1,
        )
        style.map("Calc.TButton",
                  background=[("active", "#dedede"), ("pressed", "#d2d2d2")])

        style.configure("Op.TButton",    font=("Segoe UI", 18), foreground="#111111", background="#dfe9ef", padding=(8,12))
        style.configure("Danger.TButton",font=("Segoe UI", 18, "bold"), foreground="#ffffff", background="#e74c3c", padding=(8,12))
        style.map("Danger.TButton",      background=[("active", "#cf3e2e")])
        style.configure("Warn.TButton",  font=("Segoe UI", 18, "bold"), foreground="#ffffff", background="#ef6c00", padding=(8,12))
        style.configure("Back.TButton",  font=("Segoe UI", 18), foreground="#ffffff", background="#6d4c41", padding=(8,12))
        style.configure("Eq.TButton",    font=("Segoe UI", 18, "bold"), foreground="#ffffff", background="#2ecc71", padding=(8,12))
        style.map("Eq.TButton",          background=[("active", "#28b765")])

        # ---- Display --------------------------------------------------------
        self.display = tk.Entry(
            self,
            font=("Cambria", 50),
            bd=0,
            bg="#205b7a",
            fg="#ffffff",
            justify="right",
            insertbackground="#ffffff",
        )
        self.display.grid(row=0, column=0, columnspan=4, padx=8, pady=8, ipady=10, sticky="nsew")

        # ---- Buttons (label, row, col[, colspan]) --------------------------
        self.BUTTONS = [
            ("C",  1, 0), ("CE", 1, 1), ("⌫", 1, 2), ("÷",  1, 3),
            ("7",  2, 0), ("8",  2, 1), ("9",  2, 2), ("×",  2, 3),
            ("4",  3, 0), ("5",  3, 1), ("6",  3, 2), ("−",  3, 3),
            ("1",  4, 0), ("2",  4, 1), ("3",  4, 2), ("+",  4, 3),
            ("0",  5, 0), (".",  5, 1), ("=",  5, 2, 2),
        ]

        for btn in self.BUTTONS:
            text, row, col = btn[0], btn[1], btn[2]
            colspan = btn[3] if len(btn) == 4 else 1

            # pick a style per button
            if text == "C":
                style_name = "Danger.TButton"
            elif text == "CE":
                style_name = "Warn.TButton"
            elif text == "⌫":
                style_name = "Back.TButton"
            elif text == "=":
                style_name = "Eq.TButton"
            elif text in {"÷", "×", "−", "+"}:
                style_name = "Op.TButton"
            else:
                style_name = "Calc.TButton"

            b = ttk.Button(
                self,
                text=text,
                style=style_name,
                command=lambda char=text: self.on_button_click(char)
            )
            b.grid(row=row, column=col, columnspan=colspan,
                   padx=4, pady=4, sticky="nsew")

        # Responsive grid
        for r in range(6):  # rows 0..5
            self.rowconfigure(r, weight=1)
        for c in range(4):
            self.columnconfigure(c, weight=1)

        # ---- Keyboard shortcuts --------------------------------------------
        self.bind("<Return>",    lambda e: self.on_button_click("="))
        self.bind("<KP_Enter>",  lambda e: self.on_button_click("="))
        self.bind("<Escape>",    lambda e: self.on_button_click("C"))
        self.bind("<BackSpace>", lambda e: self.on_button_click("⌫"))

        for key in "0123456789.+-*/()":
            self.bind(key, lambda e, k=key: self._type_key(k))
        self.bind("x", lambda e: self._type_key("*"))
        self.bind("X", lambda e: self._type_key("*"))
        self.bind("÷", lambda e: self._type_key("/"))

    # ---- UI Handlers --------------------------------------------------------
    def on_button_click(self, char: str):
        if char == "C":
            self.expression = ""
            self.update_display()
            return

        if char == "CE":
            self.expression = re.sub(r"\d+(\.\d+)?\s*$", "", self.expression)
            self.update_display()
            return

        if char == "⌫":
            self.expression = self.expression[:-1]
            self.update_display()
            return

        if char == "=":
            self.evaluate_expression()
            return

        if char in self.SYMBOL_MAP:
            char = self.SYMBOL_MAP[char]

        self.expression += char
        self.update_display()

    def _type_key(self, k: str):
        self.expression += k
        self.update_display()

    def update_display(self):
        self.display.delete(0, tk.END)
        self.display.insert(0, self.expression)

    # ---- Evaluation ---------------------------------------------------------
    def evaluate_expression(self):
        expr = self.expression.strip()
        try:
            if not expr or not MATH_ALLOWED.match(expr):
                raise ValueError("Invalid characters")
            result = eval(expr)
            self.expression = self._fmt(result)
            self.update_display()
        except ZeroDivisionError:
            messagebox.showerror("Error", "Division by zero")
            self.expression = ""
            self.update_display()
        except Exception:
            messagebox.showerror("Error", "Invalid expression")
            self.expression = ""
            self.update_display()

    def _fmt(self, n):
        try:
            return str(int(n)) if float(n) == int(n) else str(n)
        except Exception:
            return str(n)


if __name__ == "__main__":
    app = Calculator()
    app.mainloop()
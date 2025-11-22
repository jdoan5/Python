import tkinter as tk
import re
from tkinter import messagebox

# Very small safety gate for eval: digits, dot, + - * / ( ) and whitespace only
MATH_ALLOWED = re.compile(r"^[0-9\.\+\-\*\/\(\)\s]+$")

class Calculator(tk.Tk):
    SYMBOL_MAP = {"÷": "/", "×": "*", "−": "-"}

    def __init__(self):
        super().__init__()
        self.title("Python Calculator")
        self.resizable(True, True)
        self.expression = ""

        # ---- Display --------------------------------------------------------
        self.display = tk.Entry(
            self,
            font=("Cambria", 50),
            bd=0,
            bg="#205b7a",    # display background
            fg="#ffffff",    # display text color
            justify="right",
            insertbackground="#ffffff"
        )
        self.display.grid(row=0, column=0, columnspan=4, padx=8, pady=8, ipady=10, sticky="nsew")

        # ---- Buttons (label, row, col[, colspan]) --------------------------
        self.BUTTONS = [
            ("C",  1, 0), ("CE", 1, 1), ("⌫", 1, 2), ("÷",  1, 3),
            ("7",  2, 0), ("8",  2, 1), ("9",  2, 2), ("×",  2, 3),
            ("4",  3, 0), ("5",  3, 1), ("6",  3, 2), ("−",  3, 3),
            ("1",  4, 0), ("2",  4, 1), ("3",  4, 2), ("+",  4, 3),
            ("0",  5, 0), (".",  5, 1), ("=",  5, 2, 2),  # "=" spans two columns
        ]

        for btn in self.BUTTONS:
            text, row, col = btn[0], btn[1], btn[2]
            colspan = btn[3] if len(btn) == 4 else 1

            # Colors (macOS may ignore bg, but fg is applied)
            is_eq  = (text == "=")
            is_c   = (text == "C")
            is_ce  = (text == "CE")
            is_bsp = (text == "⌫")
            is_op  = text in {"÷", "×", "−", "+"}

            bg = "#444444"
            fg = "#ffffff"
            if is_eq:  bg = "#2e7d32"   # green
            elif is_c: bg = "#c62828"   # red
            elif is_ce: bg = "#ef6c00"  # orange
            elif is_bsp: bg = "#6d4c41" # brown-ish
            elif is_op: bg = "#455a64"  # bluish for ops

            b = tk.Button(
                self,
                text=text,
                font=("Segoe UI", 18),
                bg=bg, fg=fg, bd=0,
                activebackground="#666666",
                activeforeground="#ffffff",
                command=lambda char=text: self.on_button_click(char)
            )
            b.grid(row=row, column=col, columnspan=colspan,
                   padx=4, pady=4, ipadx=10, ipady=16, sticky="nsew")

        # Responsive grid
        for r in range(6):  # 0..5
            self.rowconfigure(r, weight=1)
        for c in range(4):
            self.columnconfigure(c, weight=1)

        # ---- Keyboard shortcuts --------------------------------------------
        # Enter = evaluate
        self.bind("<Return>",    lambda e: self.on_button_click("="))
        self.bind("<KP_Enter>",  lambda e: self.on_button_click("="))
        # Esc = clear all
        self.bind("<Escape>",    lambda e: self.on_button_click("C"))
        # Backspace = delete one char
        self.bind("<BackSpace>", lambda e: self.on_button_click("⌫"))

        # Direct typing for digits, dot, parentheses, and common ops
        for key in "0123456789.+-*/()":
            self.bind(key, lambda e, k=key: self._type_key(k))
        # Allow 'x' or 'X' to mean multiply
        self.bind("x", lambda e: self._type_key("*"))
        self.bind("X", lambda e: self._type_key("*"))
        # Divide key on some keyboards
        self.bind("÷", lambda e: self._type_key("/"))

    # ---- UI Handlers --------------------------------------------------------
    def on_button_click(self, char: str):
        if char == "C":
            # Clear all
            self.expression = ""
            self.update_display()
            return

        if char == "CE":
            # Clear entry: remove only the trailing number (and decimal part) if present
            self.expression = re.sub(r"\d+(\.\d+)?\s*$", "", self.expression)
            self.update_display()
            return

        if char == "⌫":
            # Backspace one char
            self.expression = self.expression[:-1]
            self.update_display()
            return

        if char == "=":
            self.evaluate_expression()
            return

        # Normalize pretty symbols to Python operators
        if char in self.SYMBOL_MAP:
            char = self.SYMBOL_MAP[char]

        # Append and refresh
        self.expression += char
        self.update_display()

    def _type_key(self, k: str):
        # Typed directly from keyboard
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

    # Pretty-print ints without trailing .0
    def _fmt(self, n):
        try:
            return str(int(n)) if float(n) == int(n) else str(n)
        except Exception:
            return str(n)


if __name__ == "__main__":
    app = Calculator()
    app.mainloop()
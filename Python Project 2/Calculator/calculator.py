import tkinter as tk
import tkinter.ttk as ttk
import re
from tkinter import messagebox

# Allow only safe math tokens for eval (simple guard)
MATH_ALLOWED = re.compile(r"^[0-9\.\+\-\*\/\(\)\s]+$")

class Calculator(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Python Calculator")
        self.resizable(True, True)  # allow window resize
        self.expression = ""

        # ----- ttk theme/styles (works consistently on macOS, Windows, Linux) -----
        self.style = ttk.Style(self)
        try:
            # 'clam' respects background colors on macOS
            self.style.theme_use("clam")
        except Exception:
            pass

        common_font = ("Helvetica", 20)

        # Base key style
        self.style.configure(
            "Key.TButton",
            background="#444444",
            foreground="#ffffff",
            padding=8,
            font=common_font,
            relief="flat"
        )
        self.style.map(
            "Key.TButton",
            background=[("active", "#555555")]
        )

        # Equals style
        self.style.configure(
            "Eq.TButton",
            background="#27ae60",
            foreground="#ffffff",
            padding=8,
            font=common_font,
            relief="flat"
        )
        self.style.map(
            "Eq.TButton",
            background=[("active", "#1f8f50")]
        )

        # Danger (Clear) style
        self.style.configure(
            "Danger.TButton",
            background="#ff3b30",
            foreground="#ffffff",
            padding=8,
            font=common_font,
            relief="flat"
        )
        self.style.map(
            "Danger.TButton",
            background=[("active", "#d62d20")]
        )

        # ----- Display -----
        self.display = tk.Entry(
            self,
            font=("Cambria", 50),
            bd=0,
            bg="#205b7a",         # display background
            fg="#ffffff",         # display text color
            justify="right",
            insertbackground="#ffffff"
        )
        self.display.grid(row=0, column=0, columnspan=4, padx=8, pady=8, ipady=10, sticky="nsew")

        # ----- Buttons: (label, row, col[, colspan]) -----
        buttons = [
            ("(", 1,0),  (")", 1,1),    ("e²", 1,2),   ("√", 1, 3),
            ("%", 2, 0), ("1/x", 2, 1), ("x²", 2, 2),  ("+/-", 2, 3),
            ("π", 3, 0), ("sin", 3, 1), ("cos", 3, 2), ("tan", 3, 3),
            ("7", 4, 0), ("8", 4, 1),   ("9", 4, 2),   ("/", 4, 3),
            ("4", 5, 0), ("5", 5, 1),   ("6", 5, 2),   ("*", 5, 3),
            ("1", 6, 0), ("2", 6, 1),   ("3", 6, 2),   ("-", 6, 3),
            ("0", 7, 0), (".", 7, 1),   ("C", 7, 2),   ("+", 7, 3),
            ("=", 8, 0, 4)
        ]

        for btn in buttons:
            text, row, col = btn[0], btn[1], btn[2]
            colspan = btn[3] if len(btn) == 4 else 1

            # Pick a ttk style per button
            if text == "C":
                style_name = "Danger.TButton"
            elif text == "=":
                style_name = "Eq.TButton"
            else:
                style_name = "Key.TButton"

            b = ttk.Button(
                self,
                text=text,
                style=style_name,
                command=lambda char=text: self.on_button_click(char)
            )
            b.grid(row=row, column=col, columnspan=colspan,
                   padx=4, pady=4, sticky="nsew")

        # Make grid responsive (rows 0..6)
        for i in range(7):
            self.rowconfigure(i, weight=1)
        for j in range(4):
            self.columnconfigure(j, weight=1)

    # ------------------ UI actions ------------------
    def on_button_click(self, char: str):
        if char == "C":
            self.expression = ""
            self.update_display()
        elif char == "=":
            self.evaluate_expression()
        elif char == "%":
            self.apply_percent()
        elif char == "1/x":
            self.apply_unary(lambda x: 1 / x if x != 0 else float("inf"))
        elif char == "x²":
            self.apply_unary(lambda x: x * x)
        elif char == "+/-":
            self.apply_unary(lambda x: -x)
        elif char == "π":
            self.apply_unary(lambda x: 3.141592653589793)
        else:
            self.expression += char
            self.update_display()

    def update_display(self):
        self.display.delete(0, tk.END)
        self.display.insert(0, self.expression)

    def evaluate_expression(self):
        expr = self.expression.strip()
        try:
            # very small safety gate – only allow digits, ., + - * / ( ) and whitespace
            if not expr or not MATH_ALLOWED.match(expr):
                raise ValueError("invalid chars")
            result = eval(expr)
            self.expression = str(result)
            self.update_display()
        except Exception:
            messagebox.showerror("Error", "Invalid expression")
            self.expression = ""
            self.update_display()

    # ------------------ Helpers for %, unary ops ------------------
    def _fmt(self, n: float) -> str:
        # Display integers without trailing .0
        return str(int(n)) if isinstance(n, (int, float)) and n == int(n) else str(n)

    def _replace_trailing_number(self, transform):
        """
        Find the trailing number in the expression and replace it with transform(number).
        E.g. '200+10' with negate -> '200+-10'
        """
        m = re.search(r"(\d+(?:\.\d+)?)\s*$", self.expression)
        if not m:
            return
        num = float(m.group(1))
        new_val = transform(num)
        self.expression = self.expression[:m.start(1)] + self._fmt(new_val)
        self.update_display()

    def apply_unary(self, fn):
        # apply to the last number in the expression; if empty, treat as 0
        if not self.expression.strip():
            self.expression = self._fmt(fn(0.0))
            self.update_display()
            return
        self._replace_trailing_number(fn)

    def apply_percent(self):
        """
        Context-aware % like real calculators:
          x + y%  => x + (x * y/100)
          x - y%  => x - (x * y/100)
          x * y%  => x * (y/100)
          x / y%  => x / (y/100)
        If no operator present, convert trailing number to /100.
        """
        expr = self.expression.strip()
        if not expr:
            return

        # Match simple binary form: left op right, with numeric right
        m = re.match(r"""^(?P<left>.*?)(?P<op>[+\-*/])\s*(?P<right>\d+(?:\.\d+)?)\s*$""", expr)
        if m:
            left = m.group("left").rstrip()
            op = m.group("op")
            right = float(m.group("right"))

            if op in ["+", "-"]:
                replacement = f"({left})*({right})/100"
            else:  # * or /
                replacement = f"({right})/100"

            self.expression = f"{left}{op}{replacement}"
            self.update_display()
            return

        # Fallback: just divide the trailing number by 100
        self._replace_trailing_number(lambda x: x / 100.0)


if __name__ == "__main__":
    app = Calculator()
    app.mainloop()
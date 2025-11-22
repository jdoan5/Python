import tkinter as tk
import re
from tkinter import messagebox

class Calculator(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Python Calculator")
        self.resizable(True, True) #option to resize windows

        self.expression = ""

        #calculator display screen
        self.display = tk.Entry(
            self,
            font=("Cambria", 50),
            bd=0,
            bg="#205b7a", #the result display screen
            fg="#ffffff",
            justify="right",
            insertbackground="#ffffff"
        )
        self.display.grid(row=0, column=0, columnspan=4, padx=8, pady=8, ipady=10, sticky="nsew")

        #button layout display
        buttons = [
            ("%", 1, 0), ("1/x", 1, 1), ("x²", 1, 2), ("+/-", 1, 3),
            ("7", 2, 0), ("8", 2, 1), ("9", 2, 2), ("/", 2, 3),
            ("4", 3, 0), ("5", 3, 1), ("6", 3, 2), ("*", 3, 3),
            ("1", 4, 0), ("2", 4, 1), ("3", 4, 2), ("-", 4, 3),
            ("0", 5, 0), (".", 5, 1), ("C", 5, 2), ("+", 5, 3),
            ("=", 6, 0, 4)
        ]

        for btn in buttons:
            text, row, col = btn[0], btn[1], btn[2]
            colspan = btn[3] if len(btn) == 4 else 1

            b = tk.Button(
                self,
                text=text,
                font=("Segoe UI", 16),
                bg="#333333" if text not in ("C", "=") else ("#c0392b" if text == "C" else "#27ae60"),
                fg="#ffffff",
                bd=0,
                activebackground="#555555",
                activeforeground="#ffffff",
                command=lambda char=text: self.on_button_click(char)
            )
            b.grid(row=row, column=col, columnspan=colspan,
                   padx=4, pady=4, ipadx=10, ipady=10, sticky="nsew")

        for i in range(6):
            self.rowconfigure(i, weight=1)
        for j in range(4):
            self.columnconfigure(j, weight=1)

    def on_button_click(self, char: str):
        if char == "C":
            self.expression = ""
            self.update_display()
        elif char == "=":
            self.evaluate_expression()
        elif char == "%":
            self.apply_percent()  # <-- use your helper
        elif char == "1/x":
            self.apply_unary(lambda x: 1 / x if x != 0 else float('inf'))
        elif char == "x²":
            self.apply_unary(lambda x: x * x)
        elif char == "+/-":
            self.apply_unary(lambda x: -x)
        else:
            self.expression += char
            self.update_display()

    def update_display(self):
        self.display.delete(0, tk.END)
        self.display.insert(0, self.expression)


    def evaluate_expression(self):
        try:
            result = eval(self.expression)
            self.expression = str(result)
            self.update_display()
        except Exception:
            messagebox.showerror("Error", "Invalid expression")
            self.expression = ""
            self.update_display()

    #Calcuation %, unary ops
    def _fmt(self, n: float) -> str:
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
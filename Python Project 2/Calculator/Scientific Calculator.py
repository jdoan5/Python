# Scientific-ish Calculator (macOS-safe: visible buttons via ttk)
import tkinter as tk
import re
from tkinter import messagebox, ttk

MATH_ALLOWED = re.compile(r"^[0-9\.\+\-\*\/\(\)\s]+$")

class Calculator(tk.Tk):
    SYMBOL_MAP = {"÷": "/", "×": "*", "−": "-"}

    def __init__(self):
        super().__init__()
        self.title("Scientific Calculator ")
        self.resizable(True, True)
        self.expression = ""

        # --- Display ---------------------------------------------------------
        self.display = tk.Entry(
            self, font=("Cambria", 50), bd=0,
            bg="#205b7a", fg="#ffffff", justify="right", insertbackground="#ffffff" #background display screen
        )
        self.display.grid(row=0, column=0, columnspan=8, padx=8, pady=8, ipady=10, sticky="nsew")

        # --- ttk styles (use a theme that honors colors on macOS) ------------
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("Calc.TButton",   font=("Segoe UI", 18), foreground="#111111", padding=(8, 12))
        style.configure("Op.TButton",     font=("Segoe UI", 18), foreground="#111111",
                        background="#dfe9ef", relief="flat")
        style.configure("Danger.TButton", font=("Segoe UI", 18, "bold"), foreground="#ffffff",
                        background="#e74c3c", relief="flat")
        style.configure("Warn.TButton",   font=("Segoe UI", 18, "bold"), foreground="#ffffff",
                        background="#ef6c00", relief="flat")
        style.configure("Back.TButton",   font=("Segoe UI", 18), foreground="#ffffff",
                        background="#6d4c41", relief="flat")
        style.configure("Eq.TButton",     font=("Segoe UI", 18, "bold"), foreground="#ffffff",
                        background="#2ecc71", relief="flat")
        style.map("Eq.TButton", background=[("active", "#28b765")])

        # --- Buttons ---------------------------------------------------------
        self.BUTTONS = [
            ("(",  1, 0),     (")", 1, 1),     ("C", 1, 2),     ("CE", 1, 3),    ("⌫",  1, 4),
            ("%",  2, 0),     ("1/x", 2, 1),   ("√", 2, 2),     ("π", 2, 3),      ("Rand", 2, 4),
            ("sin", 3, 0),    ("cos", 3, 1),   ("tan", 3, 2),   ("x²", 3, 3),     ("x³", 3, 4),
            ("sin⁻¹", 4, 0),  ("cos⁻¹", 4, 1), ("tan⁻¹", 4,2),  ("x!", 4, 3), ("÷", 4, 4),
            ("e",  5, 0),      ("7", 5, 1),     ("8",  5, 2),    ("9",  5, 3),    ("×", 5, 4),
            ("ln",  6, 0),      ("4", 6, 1),     ("5",  6, 2),    ("6",  6, 3),    ("-", 6, 4),
            ("log₁₀",  7, 0),      ("1", 7, 1),     ("2",  7, 2),    ("3",  7, 3),    ("+",7,4),
            ("+/-", 8, 0),    (".", 8, 1),     ("0", 8, 2),     ("Ans", 8, 3),   ("=", 8, 4),
        ]

        for data in self.BUTTONS:
            text, row, col = data[0], data[1], data[2]
            colspan = data[3] if len(data) == 4 else 1

            # choose a ttk style per key
            if text == "=":
                sty = "Eq.TButton"
            elif text == "C":
                sty = "Danger.TButton"
            elif text == "CE":
                sty = "Warn.TButton"
            elif text == "⌫":
                sty = "Back.TButton"
            elif text in {"÷", "×", "-", "+"}:
                sty = "Op.TButton"
            else:
                sty = "Calc.TButton"

            ttk.Button(
                self,
                text=text,
                style=sty,
                command=lambda ch=text: self.on_button_click(ch)
            ).grid(row=row, column=col, columnspan=colspan,
                   padx=4, pady=4, ipadx=10, ipady=16, sticky="nsew")

        # Grid weights
        for r in range(6):
            self.rowconfigure(r, weight=1)
        for c in range(4):
            self.columnconfigure(c, weight=1)

        # --- Keyboard bindings ----------------------------------------------
        self.bind("<Return>",    lambda e: self.on_button_click("="))
        self.bind("<KP_Enter>",  lambda e: self.on_button_click("="))
        self.bind("<Escape>",    lambda e: self.on_button_click("C"))
        self.bind("<BackSpace>", lambda e: self.on_button_click("⌫"))
        for key in "0123456789.+-*/()":
            self.bind(key, lambda e, k=key: self._type_key(k))
        self.bind("x", lambda e: self._type_key("*"))
        self.bind("X", lambda e: self._type_key("*"))
        self.bind("÷", lambda e: self._type_key("/"))

    # --- Handlers ------------------------------------------------------------
    def on_button_click(self, char: str):
        if char == "C":
            self.expression = ""
            return self.update_display()
        if char == "CE":
            self.expression = re.sub(r"\d+(\.\d+)?\s*$", "", self.expression)
            return self.update_display()
        if char == "⌫":
            self.expression = self.expression[:-1]
            return self.update_display()
        if char == "=":
            return self.evaluate_expression()
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

    # --- Eval ---------------------------------------------------------------
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
    Calculator().mainloop()
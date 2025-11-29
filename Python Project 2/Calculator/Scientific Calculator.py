# Extend version of Simple Calculator with more scientific functions
# A brief explanation of each line

#import Python build-in GUI library tkinter
import tkinter as tk
import math #import standard math module like cos, cos, square root etc)

class ScientificCalculator(tk.Tk): # Define a new class called ScientificCalculator inherits from tk.Tk
    def __init__(self):
        super().__init__()

        self.title("Scientific Calculator") # Set the title of the window
        self.configure(bg="#222222") # Set the background color of the window to a dark gray (#222222)

        self.expression = tk.StringVar() # Create a tk.StringVar to store the expression like (1+1 = 2)

        #---Display-------
        display = tk.Entry(
            self,
            textvariable=self.expression,
            font=("Segoe UI", 18),
            bd=0,
            relief="flat",
            justify="right", #justify = right align the calculator displays
            bg="#0d3b66", #bg = dark blue background for the display
            fg="white",
            insertbackground="white",
        )

        display.grid(row=0, column=0, columnspan=5, sticky="nsew", padx=4, pady=4, ipady=16)

        # --- Button layout (labels only) ---
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

        # --- Creating each button ---
        for r, row in enumerate(buttons, start=1):
            for c, label in enumerate(row):
                btn = tk.Button(
                    self,
                    text=label,
                    font=("Segoe UI", 14),
                    bd=0,
                    relief="flat",
                    bg=self._bg_for(label),
                    fg="white" if label in {"C", "CE", "=", "⌫"} else "black",
                    command=lambda v=label: self.on_button(v),
                    )
                btn.grid(row=r, column=c, sticky="nsew", padx=1, pady=1, ipadx=6, ipady=12)

                # make grid responsive
                for col in range(5):
                    self.grid_columnconfigure(col, weight=1) # each column should expand equally when the window is resized
                for row in range(1, 9):
                    self.grid_rowconfigure(row, weight=1)

                # store last answer for "Ans" button to recall it
                self.last_ans = None

        # --------------------------------------------------------------------- #
        # styling helper
        def _bg_for(self, label: str) -> str:
            if label == "=":
                return "#4caf50"  # green
            if label in {"C", "CE"}:
                return "#e74c3c"  # red
            if label == "⌫":
                return "#6d4c41"  # brown
            if label in {"÷", "×", "−", "+", "%"}:
                return "#e8f2ff"  # light blue
            return "#f2f0eb"  # light gray for others

        # button behavior
        def on_button(self, label: str):
            if label == "C": # clear display
                self.expression.set("")
                self.last_answer = None
                return
            if label == "CE": # clear last digit
                self.expression.set("")
                return
            if label == "⌫": # backspace
                self.expressin.set(self.expression.get()[:-1])
                return
            if label == "=": # to evaluate the expression
                self._calculate()
                return
            if label == "+/-": # toggle sign
                expr = self.expression.get()
                if expr.starswith("-"):
                        self.expression.set(expr[1:])
                else:
                        self.expression.set("-" + expr if expr else "-")
                return
            if label == "Ans" and self.last_answer is not None: #if there is a stored last_answer, recall it
                self.expression.set(self.expression.get() + str(self.last_answer))
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
                        # simple “Rand” inserts a random number 0–1
                        "Rand": "rand()",
            }
            if label in mapping:
                return mapping[label]
            # digits, parentheses, dot, %, +, etc. are used as-is
            return label

    # safe evaluation environment
    # env = a dictionary of functions that can be called by eval()
    # sin, cos, tan = math functions
    # pi, e = constants
    # rand() = a function that returns a random number (0, 1)
    # ln = math.log (natural log)
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
                    "ln": math.log,  # natural log
                    "log10": math.log10,
                    "fact": math.factorial,
                    "rand": lambda: math.random() if hasattr(math, "random") else 0.0,
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
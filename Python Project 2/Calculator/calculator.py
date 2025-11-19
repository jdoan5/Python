import platform
import tkinter as tk
from tkinter import messagebox

IS_MAC = platform.system() == "Darwin"

BTN_TEXT_COLOR = "#000000" if IS_MAC else "#ffffff"   # black on mac, white elsewhere
BTN_BG_NORMAL  = "#333333" if not IS_MAC else "#f0f0f0"
BTN_BG_CLEAR   = "#c0392b" if not IS_MAC else "#f0f0f0"
BTN_BG_EQUAL   = "#27ae60" if not IS_MAC else "#f0f0f0"

class Calculator(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Python Calculator")
        self.resizable(False, False)

        self.expression = ""

        self.display = tk.Entry(
            self,
            font=("Segoe UI", 20),
            bd=0,
            bg="green", #the result display screen
            fg="#ffffff",
            justify="right",
            insertbackground="#ffffff"
        )
        self.display.grid(row=0, column=0, columnspan=4, padx=8, pady=8, ipady=10, sticky="nsew")

        buttons = [
            ("7", 1, 0), ("8", 1, 1), ("9", 1, 2), ("/", 1, 3),
            ("4", 2, 0), ("5", 2, 1), ("6", 2, 2), ("*", 2, 3),
            ("1", 3, 0), ("2", 3, 1), ("3", 3, 2), ("-", 3, 3),
            ("0", 4, 0), (".", 4, 1), ("C", 4, 2), ("+", 4, 3),
            ("=", 5, 0, 4)
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

if __name__ == "__main__":
    app = Calculator()
    app.mainloop()
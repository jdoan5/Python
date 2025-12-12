import tkinter as tk
from tkinter import ttk, messagebox


class UnitApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Unit Converter App")
        self.geometry("480x320")
        self.resizable(False, False)

        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)

        self.frames = {}
        for Page in (MainMenu, TemperaturePage, DataSizePage):
            name = Page.__name__
            frame = Page(parent=container, controller=self)
            self.frames[name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("MainMenu")

    def show_frame(self, name: str):
        self.frames[name].tkraise()


class MainMenu(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        title = ttk.Label(self, text="Unit Converter Hub",
                          font=("Segoe UI", 18, "bold"))
        subtitle = ttk.Label(self, text="Choose a converter:",
                             font=("Segoe UI", 11))

        btn_temp = ttk.Button(
            self,
            text="Temperature Converter",
            command=lambda: controller.show_frame("TemperaturePage")
        )
        btn_data = ttk.Button(
            self,
            text="Data Size Converter",
            command=lambda: controller.show_frame("DataSizePage")
        )

        title.pack(pady=(40, 8))
        subtitle.pack(pady=(0, 20))
        btn_temp.pack(pady=8, ipadx=10, ipady=4)
        btn_data.pack(pady=8, ipadx=10, ipady=4)


class TemperaturePage(ttk.Frame):
    UNITS = ["Celsius (°C)", "Fahrenheit (°F)", "Kelvin (K)"]

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        header = ttk.Label(self, text="Temperature Converter",
                           font=("Segoe UI", 14, "bold"))
        back_btn = ttk.Button(
            self, text="<– Back to Main Menu",
            command=lambda: controller.show_frame("MainMenu")
        )

        self.value_var = tk.StringVar()
        self.from_unit = tk.StringVar(value=self.UNITS[0])
        self.to_unit = tk.StringVar(value=self.UNITS[1])
        self.result_var = tk.StringVar(value="")

        value_label = ttk.Label(self, text="Value:")
        value_entry = ttk.Entry(self, textvariable=self.value_var, width=15)

        from_label = ttk.Label(self, text="From:")
        from_combo = ttk.Combobox(self, values=self.UNITS,
                                  textvariable=self.from_unit,
                                  state="readonly", width=16)

        to_label = ttk.Label(self, text="To:")
        to_combo = ttk.Combobox(self, values=self.UNITS,
                                textvariable=self.to_unit,
                                state="readonly", width=16)

        convert_btn = ttk.Button(self, text="Convert", command=self.convert)

        result_caption = ttk.Label(self, text="Result:")
        # Use tk.Label so foreground color always applies
        result_label = tk.Label(
            self,
            textvariable=self.result_var,
            font=("Segoe UI", 11, "bold"),
            fg="#ffffff",      # bright text
            bg="#333333",      # dark background for contrast
            padx=6, pady=3
        )

        header.grid(row=0, column=0, columnspan=3, pady=(20, 10), sticky="w", padx=10)
        back_btn.grid(row=0, column=3, padx=10, sticky="e")

        value_label.grid(row=1, column=0, padx=10, pady=5, sticky="e")
        value_entry.grid(row=1, column=1, pady=5, sticky="w")

        from_label.grid(row=2, column=0, padx=10, pady=5, sticky="e")
        from_combo.grid(row=2, column=1, pady=5, sticky="w")

        to_label.grid(row=3, column=0, padx=10, pady=5, sticky="e")
        to_combo.grid(row=3, column=1, pady=5, sticky="w")

        convert_btn.grid(row=4, column=0, columnspan=2, pady=15)

        result_caption.grid(row=5, column=0, padx=10, pady=(5, 0), sticky="e")
        result_label.grid(row=5, column=1, columnspan=3, pady=(5, 0), sticky="w")

        for col in range(4):
            self.grid_columnconfigure(col, weight=1)

    def convert(self):
        try:
            val = float(self.value_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter a numeric value.")
            return

        from_u = self.from_unit.get()
        to_u = self.to_unit.get()

        # normalize to Celsius
        if "Celsius" in from_u:
            c = val
        elif "Fahrenheit" in from_u:
            c = (val - 32) * 5.0 / 9.0
        elif "Kelvin" in from_u:
            c = val - 273.15
        else:
            messagebox.showerror("Error", "Unknown source unit.")
            return

        if "Celsius" in to_u:
            result = c
        elif "Fahrenheit" in to_u:
            result = c * 9.0 / 5.0 + 32
        elif "Kelvin" in to_u:
            result = c + 273.15
        else:
            messagebox.showerror("Error", "Unknown target unit.")
            return

        text = f"{val:g} {from_u} = {result:.3f} {to_u}"
        print("TEMP RESULT:", text)   # debug in terminal
        self.result_var.set(text)


class DataSizePage(ttk.Frame):
    UNITS = [
        ("Bytes (B)", 1),
        ("Kilobytes (KB)", 1024),
        ("Megabytes (MB)", 1024 ** 2),
        ("Gigabytes (GB)", 1024 ** 3),
        ("Terabytes (TB)", 1024 ** 4),
        ("Petabytes (PB)", 1024 ** 5),
        ("Exabytes (EB)", 1024 ** 6),
        ("Zettabytes (ZB)", 1024 ** 7),
        ("Yottabytes (YB)", 1024 ** 8),
        ("Kibibytes (KiB)", 1024 ** 1),
        ("Mebibytes (MiB)", 1024 ** 2),
        ("Gibibytes (GiB)", 1024 ** 3),
        ("Tebibytes (TiB)", 1024 ** 4),
        ("Pebibytes (PiB)", 1024 ** 5),
        ("Exbibytes (EiB)", 1024 ** 6),
        ("Zebibytes (ZiB)", 1024 ** 7),
        ("Yobibytes (YiB)", 1024 ** 8), 
    ]

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        header = ttk.Label(self, text="Data Size Converter",
                           font=("Segoe UI", 14, "bold"))
        back_btn = ttk.Button(
            self, text="<– Back to Main Menu",
            command=lambda: controller.show_frame("MainMenu")
        )

        self.value_var = tk.StringVar()
        self.from_unit = tk.StringVar(value=self.UNITS[0][0])
        self.to_unit = tk.StringVar(value=self.UNITS[1][0])
        self.result_var = tk.StringVar(value="")

        value_label = ttk.Label(self, text="Value:")
        value_entry = ttk.Entry(self, textvariable=self.value_var, width=18)

        unit_names = [name for name, _ in self.UNITS]

        from_label = ttk.Label(self, text="From:")
        from_combo = ttk.Combobox(self, values=unit_names,
                                  textvariable=self.from_unit,
                                  state="readonly", width=18)

        to_label = ttk.Label(self, text="To:")
        to_combo = ttk.Combobox(self, values=unit_names,
                                textvariable=self.to_unit,
                                state="readonly", width=18)

        convert_btn = ttk.Button(self, text="Convert", command=self.convert)

        result_caption = ttk.Label(self, text="Result:")
        result_label = tk.Label(
            self,
            textvariable=self.result_var,
            font=("Segoe UI", 11, "bold"),
            fg="#ffffff",
            bg="#333333",
            padx=6, pady=3
        )

        header.grid(row=0, column=0, columnspan=3, pady=(20, 10), sticky="w", padx=10)
        back_btn.grid(row=0, column=3, padx=10, sticky="e")

        value_label.grid(row=1, column=0, padx=10, pady=5, sticky="e")
        value_entry.grid(row=1, column=1, pady=5, sticky="w")

        from_label.grid(row=2, column=0, padx=10, pady=5, sticky="e")
        from_combo.grid(row=2, column=1, pady=5, sticky="w")

        to_label.grid(row=3, column=0, padx=10, pady=5, sticky="e")
        to_combo.grid(row=3, column=1, pady=5, sticky="w")

        convert_btn.grid(row=4, column=0, columnspan=2, pady=15)

        result_caption.grid(row=5, column=0, padx=10, pady=(5, 0), sticky="e")
        result_label.grid(row=5, column=1, columnspan=3, pady=(5, 0), sticky="w")

        for col in range(4):
            self.grid_columnconfigure(col, weight=1)

    def convert(self):
        try:
            val = float(self.value_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter a numeric value.")
            return

        from_factor = self._factor_for(self.from_unit.get())
        to_factor = self._factor_for(self.to_unit.get())

        if from_factor is None or to_factor is None:
            messagebox.showerror("Error", "Unknown unit.")
            return

        bytes_value = val * from_factor
        result = bytes_value / to_factor

        text = f"{val:g} {self.from_unit.get()} = {result:.3f} {self.to_unit.get()}"
        print("DATA RESULT:", text)   # debug in terminal
        self.result_var.set(text)

    def _factor_for(self, name: str):
        for unit_name, factor in self.UNITS:
            if unit_name == name:
                return factor
        return None


if __name__ == "__main__":
    app = UnitApp()
    app.mainloop()
import tkinter as tk
from tkinter import ttk, messagebox
from currencies import load_currencies

class UnitApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Unit Converter App")
        self.geometry("640x420")       # stable size on macOS
        self.resizable(False, False)

        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)

        # Pages use grid inside the same container
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for Page in (MainMenu, BodyMassIndexPage,CurrencyPage, DataSizePage, TemperaturePage):
            name = Page.__name__
            frame = Page(parent=container, controller=self)
            self.frames[name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("MainMenu")

    def show_frame(self, name: str):
        self.frames[name].tkraise()

class MainMenu(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, padding=24)
        self.controller = controller

        title = ttk.Label(self, text="Unit Converter Hub", font=("Segoe UI", 18, "bold"))
        subtitle = ttk.Label(self, text="Choose a converter:", font=("Segoe UI", 11))

        btn_bmi = ttk.Button(
            self,
            text="Body Mass Index Calculator",
            command=lambda: controller.show_frame("BodyMassIndexPage")
        )

        btn_curr = ttk.Button(
            self, text="Currency Converter",
            command=lambda: controller.show_frame("CurrencyPage")
        )
        btn_data = ttk.Button(
            self, text="Data Size Converter",
            command=lambda: controller.show_frame("DataSizePage")
        )
        btn_temp = ttk.Button(
            self, text="Temperature Converter",
            command=lambda: controller.show_frame("TemperaturePage")
        )

        title.pack(pady=(30, 6))
        subtitle.pack(pady=(0, 18))
        btn_bmi.pack(pady=8, ipadx=14, ipady=6)
        btn_curr.pack(pady=8, ipadx=14, ipady=6)
        btn_data.pack(pady=8, ipadx=14, ipady=6)
        btn_temp.pack(pady=8, ipadx=14, ipady=6)

class BodyMassIndexPage(ttk.Frame):
    UNITS = ["Feet", "Inches", "Centimeters", "Meters"]

    def __init__(self, parent, controller):
        super().__init__(parent, padding=14)
        self.controller = controller

        header = ttk.Label(self, text="BMI Converter", font=("Segoe UI", 14, "bold"))
        back_btn = ttk.Button(
            self, text="↩ Back to Main Menu",
            command=lambda: controller.show_frame("MainMenu")
        )

        self.value_var = tk.StringVar()
        self.from_unit = tk.StringVar(value=self.UNITS[0])
        self.to_unit = tk.StringVar(value=self.UNITS[1])
        self.result_var = tk.StringVar(value="")

        value_label = ttk.Label(self, text="Value:")
        value_entry = ttk.Entry(self, textvariable=self.value_var, width=20)

        from_label = ttk.Label(self, text="From:")
        from_combo = ttk.Combobox(self, values=self.UNITS, textvariable=self.from_unit, state="readonly", width=20)

        to_label = ttk.Label(self, text="To:")
        to_combo = ttk.Combobox(self, values=self.UNITS, textvariable=self.to_unit, state="readonly", width=20)

        convert_btn = ttk.Button(self, text="Convert", command=self.convert)
        swap_btn = ttk.Button(self, text="↔ Swap", command=self.swap_units)

        result_caption = ttk.Label(self, text="Result:")
        result_label = tk.Label(
            self, textvariable=self.result_var,
            font=("Segoe UI", 11, "bold"),
            fg="#ffffff", bg="#333333",
            padx=10, pady=6, anchor="w"
        )

        header.grid(row=0, column=0, columnspan=2, pady=(6, 10), sticky="w")
        back_btn.grid(row=0, column=3, pady=(6, 10), sticky="e")

        value_label.grid(row=1, column=0, padx=(0, 10), pady=6, sticky="e")
        value_entry.grid(row=1, column=1, pady=6, sticky="w")

        from_label.grid(row=2, column=0, padx=(0, 10), pady=6, sticky="e")
        from_combo.grid(row=2, column=1, pady=6, sticky="w")

        to_label.grid(row=3, column=0, padx=(0, 10), pady=6, sticky="e")
        to_combo.grid(row=3, column=1, pady=6, sticky="w")

        convert_btn.grid(row=4, column=0, pady=12, sticky="e")
        swap_btn.grid(row=4, column=1, pady=12, sticky="w")

        result_caption.grid(row=5, column=0, padx=(0, 10), pady=(10, 0), sticky="e")
        result_label.grid(row=5, column=1, columnspan=3, pady=(10, 0), sticky="we")

        for col in range(4):
            self.grid_columnconfigure(col, weight=1)

    def swap_units(self):
        a = self.from_unit.get()
        b = self.to_unit.get()
        self.from_unit.set(b)
        self.to_unit.set(a)
        self.result_var.set("")

    def convert(self):
        try:
            val = float((self.value_var.get() or "").replace(",", ""))
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter a numeric value.")
            return

        from_u = self.from_unit.get()
        to_u = self.to_unit.get()

        if "Feet" in from_u:
            c = val
        elif "Centimeters" in from_u:
            c = (val - 32) * 5.0 / 9.0
        elif "Kelvin" in from_u:
            c = val - 273.15
        else:
            messagebox.showerror("Error", "Unknown source unit.")
            return

        if "Feet" in to_u:
            result = c
        elif "Centimeters" in to_u:
            result = c * 9.0 / 5.0 + 32
        elif "Kelvin" in to_u:
            result = c + 273.15
        else:
            messagebox.showerror("Error", "Unknown target unit.")
            return

        self.result_var.set(f"{val:g} {from_u} = {result:.3f} {to_u}")

class CurrencyPage(ttk.Frame):
    """
    LOCAL/DEMO currency converter (no live rates).
    """

    # Demo rates: "1 unit of currency == X USD" (approx values; not live)
    # Add more if you want; anything not here will show "not available".
    DEMO_TO_USD = {
        "USD": 1.00,
        "EUR": 1.0 / 0.85,
        "GBP": 1.27,
        "VND": 1.0 / 26305.96,
        "JPY": 1.0 / 150.0,
        "CAD": 1.0 / 1.35,
        "AUD": 1.0 / 1.50,
        "CHF": 1.12,
        "CNY": 1.0 / 7.20,
        "HKD": 1.0 / 7.80,
        "SGD": 1.0 / 1.35,
        "INR": 1.0 / 83.0,
        "KRW": 1.0 / 1350.0,
        "MXN": 1.0 / 17.0,
        "BRL": 1.0 / 5.0,
    }

    def __init__(self, parent, controller):
        super().__init__(parent, padding=14)
        self.controller = controller

        header = ttk.Label(self, text="Currency Converter", font=("Segoe UI", 14, "bold"))
        back_btn = ttk.Button(
            self, text="↩ Back to Main Menu",
            command=lambda: controller.show_frame("MainMenu"),
        )

        # Load full currency list from your CSV
        try:
            self.currencies = load_currencies("data/currency.csv")
        except Exception as e:
            self.currencies = [("US Dollar", "USD"), ("Euro", "EUR"), ("Pound Sterling", "GBP"), ("Vietnam Dong", "VND")]
            messagebox.showwarning("Currency list not loaded",
                                   f"Could not load data/currency.csv.\n\n{e}\n\nUsing a small fallback list.")

        self.display_list = [f"{name} ({code})" for name, code in self.currencies]
        self.display_to_code = {f"{name} ({code})": code for name, code in self.currencies}

        self.value_var = tk.StringVar()
        self.from_display = tk.StringVar(value=self.display_list[0] if self.display_list else "")
        self.to_display = tk.StringVar(value=self.display_list[1] if len(self.display_list) > 1 else (self.display_list[0] if self.display_list else ""))
        self.result_var = tk.StringVar(value="")

        note = ttk.Label(
            self,
            text=(
                "Note: This is a LOCAL demo converter (no live exchange rates).\n"
            ),
            foreground="red", font=("Segoe UI", 20)
        )

        value_label = ttk.Label(self, text="Value:")
        value_entry = ttk.Entry(self, textvariable=self.value_var, width=20)

        from_label = ttk.Label(self, text="From:")
        self.from_combo = ttk.Combobox(
            self, values=self.display_list, textvariable=self.from_display,
            state="normal", width=42
        )

        to_label = ttk.Label(self, text="To:")
        self.to_combo = ttk.Combobox(
            self, values=self.display_list, textvariable=self.to_display,
            state="normal", width=42
        )

        # Lightweight “type to filter” for usability with large lists
        self.from_combo.bind("<KeyRelease>", lambda _e: self._filter_combo(self.from_combo, self.from_display))
        self.to_combo.bind("<KeyRelease>", lambda _e: self._filter_combo(self.to_combo, self.to_display))

        convert_btn = ttk.Button(self, text="Convert", command=self.convert)
        swap_btn = ttk.Button(self, text="↔ Swap", command=self.swap_units)

        result_caption = ttk.Label(self, text="Result:")
        result_label = tk.Label(
            self,
            textvariable=self.result_var,
            font=("Segoe UI", 11, "bold"),
            fg="#ffffff",
            bg="#333333",
            padx=10, pady=6,
            anchor="w"
        )

        header.grid(row=0, column=0, columnspan=2, pady=(6, 6), sticky="w")
        back_btn.grid(row=0, column=3, pady=(6, 6), sticky="e")

        note.grid(row=1, column=0, columnspan=4, pady=(0, 10), sticky="w")

        value_label.grid(row=2, column=0, padx=(0, 10), pady=6, sticky="e")
        value_entry.grid(row=2, column=1, pady=6, sticky="w")

        from_label.grid(row=3, column=0, padx=(0, 10), pady=6, sticky="e")
        self.from_combo.grid(row=3, column=1, columnspan=3, pady=6, sticky="w")

        to_label.grid(row=4, column=0, padx=(0, 10), pady=6, sticky="e")
        self.to_combo.grid(row=4, column=1, columnspan=3, pady=6, sticky="w")

        convert_btn.grid(row=5, column=0, pady=12, sticky="e")
        swap_btn.grid(row=5, column=1, pady=12, sticky="w")

        result_caption.grid(row=6, column=0, padx=(0, 10), pady=(10, 0), sticky="e")
        result_label.grid(row=6, column=1, columnspan=3, pady=(10, 0), sticky="we")

        for col in range(4):
            self.grid_columnconfigure(col, weight=1)

    def _filter_combo(self, combo: ttk.Combobox, var: tk.StringVar):
        """
        Filters the dropdown values based on what the user typed.
        Keeps it capped so large currency lists stay responsive.
        """
        typed = (var.get() or "").strip().lower()
        if not typed:
            combo["values"] = self.display_list
            return

        matches = [x for x in self.display_list if typed in x.lower()]
        combo["values"] = matches[:200]  # cap for performance

    def _code_from_display(self, display: str) -> str:
        if display in self.display_to_code:
            return self.display_to_code[display]
        # If user typed something not exactly matching, try a best-effort parse "(CODE)"
        if "(" in display and display.endswith(")"):
            return display.split("(")[-1][:-1].strip().upper()
        raise ValueError("Select a valid currency from the list.")

    def swap_units(self):
        a = self.from_display.get()
        b = self.to_display.get()
        self.from_display.set(b)
        self.to_display.set(a)
        self.result_var.set("")

    def _rate(self, from_code: str, to_code: str) -> float:
        # Cross-rate using USD as the anchor:
        # 1 FROM = (USD_per_FROM / USD_per_TO) TO
        if from_code not in self.DEMO_TO_USD:
            raise RuntimeError(f"No demo rate available for {from_code}")
        if to_code not in self.DEMO_TO_USD:
            raise RuntimeError(f"No demo rate available for {to_code}")
        usd_per_from = self.DEMO_TO_USD[from_code]
        usd_per_to = self.DEMO_TO_USD[to_code]
        return usd_per_from / usd_per_to

    def convert(self):
        raw = (self.value_var.get() or "").strip()
        if not raw:
            self.result_var.set("Enter a value to convert.")
            return

        try:
            amount = float(raw.replace(",", ""))
        except ValueError:
            self.result_var.set("Invalid number.")
            return

        try:
            from_code = self._code_from_display(self.from_display.get())
            to_code = self._code_from_display(self.to_display.get())
            rate = self._rate(from_code, to_code)
            converted = amount * rate
        except Exception as e:
            self.result_var.set(f"Conversion unavailable: {e}")
            return

        self.result_var.set(f"{amount:,.2f} {from_code} = {converted:,.2f} {to_code}")

class DataSizePage(ttk.Frame):
    UNITS_DECIMAL = [
        ("Bits (bit)", 1 / 8),
        ("Bytes (B)", 1),
        ("Kilobytes (KB)", 1000),
        ("Megabytes (MB)", 1000**2),
        ("Gigabytes (GB)", 1000**3),
        ("Terabytes (TB)", 1000**4),
        ("Petabytes (PB)", 1000**5),
        ("Exabytes (EB)", 1000**6),
        ("Zettabytes (ZB)", 1000**7),
        ("Yottabytes (YB)", 1000**8),
    ]

    UNITS_BINARY = [
        ("Bits (bit)", 1 / 8),
        ("Bytes (B)", 1),
        ("Kibibytes (KiB)", 1024),
        ("Mebibytes (MiB)", 1024**2),
        ("Gibibytes (GiB)", 1024**3),
        ("Tebibytes (TiB)", 1024**4),
        ("Pebibytes (PiB)", 1024**5),
        ("Exbibytes (EiB)", 1024**6),
        ("Zebibytes (ZiB)", 1024**7),
        ("Yobibytes (YiB)", 1024**8),
    ]

    def __init__(self, parent, controller):
        super().__init__(parent, padding=14)
        self.controller = controller

        header = ttk.Label(self, text="Data Size Converter", font=("Segoe UI", 14, "bold"))
        back_btn = ttk.Button(
            self, text="↩ Back to Main Menu",
            command=lambda: controller.show_frame("MainMenu")
        )

        self.mode_var = tk.IntVar(value=0)

        mode_frame = ttk.Frame(self)
        ttk.Label(mode_frame, text="Mode:").pack(side="left", padx=(0, 8))
        ttk.Radiobutton(
            mode_frame, text="Decimal (KB/MB)",
            variable=self.mode_var, value=0,
            command=self._refresh_units
        ).pack(side="left", padx=(0, 12))
        ttk.Radiobutton(
            mode_frame, text="Binary (KiB/MiB)",
            variable=self.mode_var, value=1,
            command=self._refresh_units
        ).pack(side="left")

        self.value_var = tk.StringVar()
        self.from_unit = tk.StringVar()
        self.to_unit = tk.StringVar()
        self.result_var = tk.StringVar(value="")

        value_label = ttk.Label(self, text="Value:")
        value_entry = ttk.Entry(self, textvariable=self.value_var, width=20)

        from_label = ttk.Label(self, text="From:")
        self.from_combo = ttk.Combobox(self, state="readonly", width=20, textvariable=self.from_unit)

        to_label = ttk.Label(self, text="To:")
        self.to_combo = ttk.Combobox(self, state="readonly", width=20, textvariable=self.to_unit)

        convert_btn = ttk.Button(self, text="Convert", command=self.convert)
        swap_btn = ttk.Button(self, text="↔ Swap", command=self.swap_units)

        result_caption = ttk.Label(self, text="Result:")
        result_label = tk.Label(
            self, textvariable=self.result_var,
            font=("Segoe UI", 11, "bold"),
            fg="#ffffff", bg="#333333",
            padx=10, pady=6, anchor="w"
        )

        header.grid(row=0, column=0, columnspan=2, pady=(6, 6), sticky="w")
        back_btn.grid(row=0, column=3, pady=(6, 6), sticky="e")

        mode_frame.grid(row=1, column=0, columnspan=4, pady=(0, 10), sticky="w")

        value_label.grid(row=2, column=0, padx=(0, 10), pady=6, sticky="e")
        value_entry.grid(row=2, column=1, pady=6, sticky="w")

        from_label.grid(row=3, column=0, padx=(0, 10), pady=6, sticky="e")
        self.from_combo.grid(row=3, column=1, pady=6, sticky="w")

        to_label.grid(row=4, column=0, padx=(0, 10), pady=6, sticky="e")
        self.to_combo.grid(row=4, column=1, pady=6, sticky="w")

        convert_btn.grid(row=5, column=0, pady=12, sticky="e")
        swap_btn.grid(row=5, column=1, pady=12, sticky="w")

        result_caption.grid(row=6, column=0, padx=(0, 10), pady=(10, 0), sticky="e")
        result_label.grid(row=6, column=1, columnspan=3, pady=(10, 0), sticky="we")

        for col in range(4):
            self.grid_columnconfigure(col, weight=1)

        self._refresh_units()

    @staticmethod
    def fmt(x: float) -> str:
        if x == 0:
            return "0"
        ax = abs(x)
        if 0.001 <= ax < 1e6:
            if ax < 1:
                return f"{x:.9f}".rstrip("0").rstrip(".")
            return f"{x:,.6f}".rstrip("0").rstrip(".")
        sci = f"{x:.6g}"
        if ax < 1:
            dec = f"{x:.12f}".rstrip("0").rstrip(".")
        else:
            dec = f"{x:,.0f}"
        return f"{sci} ({dec})"

    def _active_units(self):
        return self.UNITS_BINARY if self.mode_var.get() == 1 else self.UNITS_DECIMAL

    def _refresh_units(self):
        units = self._active_units()
        names = [name for name, _ in units]
        self.from_combo["values"] = names
        self.to_combo["values"] = names
        if self.from_unit.get() not in names:
            self.from_unit.set(names[1] if len(names) > 1 else names[0])
        if self.to_unit.get() not in names:
            self.to_unit.set(names[2] if len(names) > 2 else names[0])
        self.result_var.set("")

    def swap_units(self):
        a = self.from_unit.get()
        b = self.to_unit.get()
        self.from_unit.set(b)
        self.to_unit.set(a)
        self.result_var.set("")

    def _factor_for(self, name: str):
        for unit_name, factor in self._active_units():
            if unit_name == name:
                return factor
        return None

    def convert(self):
        try:
            val = float((self.value_var.get() or "").replace(",", ""))
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
        self.result_var.set(f"{self.fmt(val)} {self.from_unit.get()} = {self.fmt(result)} {self.to_unit.get()}")

class TemperaturePage(ttk.Frame):
    UNITS = ["Celsius (°C)", "Fahrenheit (°F)", "Kelvin (K)"]

    def __init__(self, parent, controller):
        super().__init__(parent, padding=14)
        self.controller = controller

        header = ttk.Label(self, text="Temperature Converter", font=("Segoe UI", 14, "bold"))
        back_btn = ttk.Button(
            self, text="↩ Back to Main Menu",
            command=lambda: controller.show_frame("MainMenu")
        )

        self.value_var = tk.StringVar()
        self.from_unit = tk.StringVar(value=self.UNITS[0])
        self.to_unit = tk.StringVar(value=self.UNITS[1])
        self.result_var = tk.StringVar(value="")

        value_label = ttk.Label(self, text="Value:")
        value_entry = ttk.Entry(self, textvariable=self.value_var, width=20)

        from_label = ttk.Label(self, text="From:")
        from_combo = ttk.Combobox(self, values=self.UNITS, textvariable=self.from_unit, state="readonly", width=20)

        to_label = ttk.Label(self, text="To:")
        to_combo = ttk.Combobox(self, values=self.UNITS, textvariable=self.to_unit, state="readonly", width=20)

        convert_btn = ttk.Button(self, text="Convert", command=self.convert)
        swap_btn = ttk.Button(self, text="↔ Swap", command=self.swap_units)

        result_caption = ttk.Label(self, text="Result:")
        result_label = tk.Label(
            self, textvariable=self.result_var,
            font=("Segoe UI", 11, "bold"),
            fg="#ffffff", bg="#333333",
            padx=10, pady=6, anchor="w"
        )

        header.grid(row=0, column=0, columnspan=2, pady=(6, 10), sticky="w")
        back_btn.grid(row=0, column=3, pady=(6, 10), sticky="e")

        value_label.grid(row=1, column=0, padx=(0, 10), pady=6, sticky="e")
        value_entry.grid(row=1, column=1, pady=6, sticky="w")

        from_label.grid(row=2, column=0, padx=(0, 10), pady=6, sticky="e")
        from_combo.grid(row=2, column=1, pady=6, sticky="w")

        to_label.grid(row=3, column=0, padx=(0, 10), pady=6, sticky="e")
        to_combo.grid(row=3, column=1, pady=6, sticky="w")

        convert_btn.grid(row=4, column=0, pady=12, sticky="e")
        swap_btn.grid(row=4, column=1, pady=12, sticky="w")

        result_caption.grid(row=5, column=0, padx=(0, 10), pady=(10, 0), sticky="e")
        result_label.grid(row=5, column=1, columnspan=3, pady=(10, 0), sticky="we")

        for col in range(4):
            self.grid_columnconfigure(col, weight=1)

    def swap_units(self):
        a = self.from_unit.get()
        b = self.to_unit.get()
        self.from_unit.set(b)
        self.to_unit.set(a)
        self.result_var.set("")

    def convert(self):
        try:
            val = float((self.value_var.get() or "").replace(",", ""))
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter a numeric value.")
            return

        from_u = self.from_unit.get()
        to_u = self.to_unit.get()

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

        self.result_var.set(f"{val:g} {from_u} = {result:.3f} {to_u}")

if __name__ == "__main__":
    app = UnitApp()
    app.mainloop()
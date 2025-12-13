import tkinter as tk
import json
import threading
import time
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from tkinter import ttk, messagebox


class UnitApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Unit Converter App")
        self.resizable(False, False)

        # Give a safe starting size (macOS can show blank if initial size is odd)
        self.geometry("520x360")

        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)

        # IMPORTANT: make the container a real grid parent for pages
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # Prevent pack from shrinking container to nothing
        container.pack_propagate(False)

        self.frames = {}
        for Page in (MainMenu, CurrencyPage, DataSizePage, TemperaturePage):
            name = Page.__name__
            frame = Page(parent=container, controller=self)
            self.frames[name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("MainMenu")

    def show_frame(self, name: str):
        frame = self.frames[name]
        frame.tkraise()

        # Force layout to compute correct requested sizes before resizing
        self.update_idletasks()

        # Optional: auto-resize to the page’s requested size
        w = frame.winfo_reqwidth()
        h = frame.winfo_reqheight()

        # Guard against “0x0” requested sizes during first render
        if w < 200: w = 520
        if h < 200: h = 360

        self.geometry(f"{w + 30}x{h + 30}")

class MainMenu(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        title = ttk.Label(self, text="Unit Converter Hub", font=("Segoe UI", 18, "bold"))
        subtitle = ttk.Label(self, text="Choose a converter:", font=("Segoe UI", 11))

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

        title.pack(pady=(40, 8))
        subtitle.pack(pady=(0, 18))
        btn_curr.pack(pady=8, ipadx=14, ipady=6)
        btn_data.pack(pady=8, ipadx=14, ipady=6)
        btn_temp.pack(pady=8, ipadx=14, ipady=6)

        # Give MainMenu a sensible requested size
        self.configure(padding=20)


class CurrencyPage(ttk.Frame):
    """
    No-API-key live exchange rates using Frankfurter (ECB-based):
      https://www.frankfurter.app/docs/

    IMPORTANT:
      Frankfurter (ECB) does NOT support every currency as a 'from' base (VND often not supported as base).
      So we always fetch EUR-based rates and compute cross rates:
        rate(from->to) = (EUR->to) / (EUR->from)
    """

    CURRENCIES = [
        ("Vietnam Dong", "VND"),
        ("US Dollar", "USD"),
        ("Euro", "EUR"),
        ("Pound Sterling", "GBP"),
    ]

    API_URL = "https://open.er-api.com/v6/latest/USD"

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(padding=12)

        header = ttk.Label(self, text="Currency Converter", font=("Segoe UI", 14, "bold"))
        back_btn = ttk.Button(
            self, text="↩ Back to Main Menu",
            command=lambda: controller.show_frame("MainMenu")
        )

        # ---- State ----
        self.value_var = tk.StringVar()
        self.from_unit = tk.StringVar(value=self.CURRENCIES[0][0])
        self.to_unit = tk.StringVar(value=self.CURRENCIES[1][0])

        self.result_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Rates not loaded yet.")
        self.last_updated_var = tk.StringVar(value="Last updated: —")

        # Cache EUR-based rates
        self._eur_cache = None  # {"rates": {...}, "date": "...", "ts": epoch}
        self.cache_ttl_seconds = 60 * 30  # 30 minutes

        # ---- Widgets ----
        value_label = ttk.Label(self, text="Value:")
        value_entry = ttk.Entry(self, textvariable=self.value_var, width=18)

        currency_names = [name for name, _code in self.CURRENCIES]

        from_label = ttk.Label(self, text="From:")
        from_combo = ttk.Combobox(
            self, values=currency_names, textvariable=self.from_unit,
            state="readonly", width=20
        )

        to_label = ttk.Label(self, text="To:")
        to_combo = ttk.Combobox(
            self, values=currency_names, textvariable=self.to_unit,
            state="readonly", width=20
        )

        convert_btn = ttk.Button(self, text="Convert", command=self.convert)
        swap_btn = ttk.Button(self, text="↔ Swap", command=self.swap_units)
        refresh_btn = ttk.Button(self, text="↻ Refresh Rates", command=self.refresh_rates)

        result_caption = ttk.Label(self, text="Result:")
        result_label = tk.Label(
            self, textvariable=self.result_var,
            font=("Segoe UI", 11, "bold"),
            fg="#ffffff", bg="#333333",
            padx=8, pady=5, anchor="w"
        )

        status_label = ttk.Label(self, textvariable=self.status_var)
        last_updated_label = ttk.Label(self, textvariable=self.last_updated_var)

        # ---- Layout ----
        header.grid(row=0, column=0, columnspan=2, pady=(6, 10), sticky="w")
        back_btn.grid(row=0, column=3, pady=(6, 10), sticky="e")

        value_label.grid(row=1, column=0, padx=(0, 10), pady=6, sticky="e")
        value_entry.grid(row=1, column=1, pady=6, sticky="w")

        from_label.grid(row=2, column=0, padx=(0, 10), pady=6, sticky="e")
        from_combo.grid(row=2, column=1, pady=6, sticky="w")

        to_label.grid(row=3, column=0, padx=(0, 10), pady=6, sticky="e")
        to_combo.grid(row=3, column=1, pady=6, sticky="w")

        convert_btn.grid(row=4, column=0, pady=10, sticky="e")
        swap_btn.grid(row=4, column=1, pady=10, sticky="w")
        refresh_btn.grid(row=4, column=2, pady=10, sticky="w")

        result_caption.grid(row=5, column=0, padx=(0, 10), pady=(8, 0), sticky="e")
        result_label.grid(row=5, column=1, columnspan=3, pady=(8, 0), sticky="we")

        status_label.grid(row=6, column=0, columnspan=4, pady=(12, 0), sticky="w")
        last_updated_label.grid(row=7, column=0, columnspan=4, pady=(2, 0), sticky="w")

        for col in range(4):
            self.grid_columnconfigure(col, weight=1)

        # Auto-fetch once
        self.refresh_rates()

    # ---------- Helpers ----------
    def _code_for_name(self, name: str) -> str:
        for n, code in self.CURRENCIES:
            if n == name:
                return code
        raise ValueError("Unknown currency.")

    def swap_units(self):
        a = self.from_unit.get()
        b = self.to_unit.get()
        self.from_unit.set(b)
        self.to_unit.set(a)
        self.result_var.set("")

    # ---------- Live rates (EUR base) ----------
    def refresh_rates(self):
        self.status_var.set("Fetching live rates (EUR base)...")
        t = threading.Thread(target=self._fetch_rates_thread, daemon=True)
        t.start()

    def _fetch_rates_thread(self):
        try:
            payload = self._fetch_eur_payload()
            self._eur_cache = {
                "rates": payload["rates"],
                "date": payload.get("date", ""),
                "ts": time.time(),
            }
            self.after(0, self._on_rates_loaded)
        except Exception as exc:
            msg = f"Rate fetch failed: {exc}"
            # capture msg (NOT exc) to avoid the free-variable lambda bug
            self.after(0, lambda m=msg: self.status_var.set(m))

    def _on_rates_loaded(self):
        date = (self._eur_cache or {}).get("date") or "unknown date"
        self.status_var.set("Live rates loaded.")
        self.last_updated_var.set(f"Last updated: {date}")

    def _fetch_eur_payload(self) -> dict:
        req = Request(self.API_URL, headers={"User-Agent": "UnitConverterApp/1.0"})
        try:
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError) as e:
            raise RuntimeError(f"network error ({e})")
        except Exception as e:
            raise RuntimeError(f"unexpected error ({e})")

        if "rates" not in data:
            raise RuntimeError("invalid API response")
        return data

    def _get_eur_rates(self) -> dict:
        cached = self._eur_cache
        if cached and (time.time() - cached["ts"] < self.cache_ttl_seconds):
            return cached["rates"]

        payload = self._fetch_eur_payload()
        self._eur_cache = {
            "rates": payload["rates"],
            "date": payload.get("date", ""),
            "ts": time.time(),
        }
        return self._eur_cache["rates"]

    def _cross_rate(self, from_code: str, to_code: str) -> float:
        rates = self._get_eur_rates()

        def eur_to(code: str) -> float:
            if code == "EUR":
                return 1.0
            if code not in rates:
                raise RuntimeError(f"Rate not available for {code} (ECB list)")
            return float(rates[code])

        eur_to_from = eur_to(from_code)
        eur_to_to = eur_to(to_code)
        return eur_to_to / eur_to_from

    def convert(self):
        raw = (self.value_var.get() or "").strip()
        if not raw:
            self.result_var.set("Enter a value to convert.")
            return

        cleaned = raw.replace(",", "")
        try:
            amount = float(cleaned)
        except ValueError:
            self.result_var.set("Invalid number.")
            return

        from_code = self._code_for_name(self.from_unit.get())
        to_code = self._code_for_name(self.to_unit.get())

        try:
            rate = self._cross_rate(from_code, to_code)
        except Exception as exc:
            self.result_var.set(f"Could not load rate: {exc}")
            return

        converted = amount * rate
        self.result_var.set(f"{amount:,.2f} {from_code} = {converted:,.2f} {to_code}")


class DataSizePage(ttk.Frame):
    # Decimal (SI): KB/MB/GB use 1000
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

    # Binary (IEC): KiB/MiB/GiB use 1024
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
        super().__init__(parent)
        self.controller = controller
        self.configure(padding=12)

        header = ttk.Label(self, text="Data Size Converter", font=("Segoe UI", 14, "bold"))
        back_btn = ttk.Button(
            self, text="↩ Back to Main Menu",
            command=lambda: controller.show_frame("MainMenu")
        )

        # 0 = Decimal, 1 = Binary
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
        value_entry = ttk.Entry(self, textvariable=self.value_var, width=18)

        from_label = ttk.Label(self, text="From:")
        self.from_combo = ttk.Combobox(self, state="readonly", width=18, textvariable=self.from_unit)

        to_label = ttk.Label(self, text="To:")
        self.to_combo = ttk.Combobox(self, state="readonly", width=18, textvariable=self.to_unit)

        convert_btn = ttk.Button(self, text="Convert", command=self.convert)
        swap_btn = ttk.Button(self, text="↔ Swap", command=self.swap_units)

        result_caption = ttk.Label(self, text="Result:")
        result_label = tk.Label(
            self, textvariable=self.result_var,
            font=("Segoe UI", 11, "bold"),
            fg="#ffffff", bg="#333333",
            padx=8, pady=5, anchor="w"
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

        convert_btn.grid(row=5, column=0, pady=10, sticky="e")
        swap_btn.grid(row=5, column=1, pady=10, sticky="w")

        result_caption.grid(row=6, column=0, padx=(0, 10), pady=(10, 0), sticky="e")
        result_label.grid(row=6, column=1, columnspan=3, pady=(10, 0), sticky="we")

        for col in range(4):
            self.grid_columnconfigure(col, weight=1)

        self._refresh_units()

    @staticmethod
    def fmt(x: float) -> str:
        """
        Readable numeric formatting.
        - Normal range: plain decimal
        - Extreme range: show BOTH scientific and decimal, e.g.:
            1e-09 (0.000000001)
            1e+06 (1,000,000)
        """
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
            self.from_unit.set(names[1] if len(names) > 1 else names[0])  # default Bytes-ish
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

        self.result_var.set(
            f"{self.fmt(val)} {self.from_unit.get()} = {self.fmt(result)} {self.to_unit.get()}"
        )


class TemperaturePage(ttk.Frame):
    UNITS = ["Celsius (°C)", "Fahrenheit (°F)", "Kelvin (K)"]

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(padding=12)

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
        value_entry = ttk.Entry(self, textvariable=self.value_var, width=18)

        from_label = ttk.Label(self, text="From:")
        from_combo = ttk.Combobox(
            self, values=self.UNITS, textvariable=self.from_unit,
            state="readonly", width=18
        )

        to_label = ttk.Label(self, text="To:")
        to_combo = ttk.Combobox(
            self, values=self.UNITS, textvariable=self.to_unit,
            state="readonly", width=18
        )

        convert_btn = ttk.Button(self, text="Convert", command=self.convert)
        swap_btn = ttk.Button(self, text="↔ Swap", command=self.swap_units)

        result_caption = ttk.Label(self, text="Result:")
        result_label = tk.Label(
            self, textvariable=self.result_var,
            font=("Segoe UI", 11, "bold"),
            fg="#ffffff", bg="#333333",
            padx=8, pady=5, anchor="w"
        )

        header.grid(row=0, column=0, columnspan=2, pady=(6, 10), sticky="w")
        back_btn.grid(row=0, column=3, pady=(6, 10), sticky="e")

        value_label.grid(row=1, column=0, padx=(0, 10), pady=6, sticky="e")
        value_entry.grid(row=1, column=1, pady=6, sticky="w")

        from_label.grid(row=2, column=0, padx=(0, 10), pady=6, sticky="e")
        from_combo.grid(row=2, column=1, pady=6, sticky="w")

        to_label.grid(row=3, column=0, padx=(0, 10), pady=6, sticky="e")
        to_combo.grid(row=3, column=1, pady=6, sticky="w")

        convert_btn.grid(row=4, column=0, pady=10, sticky="e")
        swap_btn.grid(row=4, column=1, pady=10, sticky="w")

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

        self.result_var.set(f"{val:g} {from_u} = {result:.3f} {to_u}")


if __name__ == "__main__":
    app = UnitApp()
    app.mainloop()
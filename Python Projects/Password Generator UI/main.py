#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import secrets, string, math
from datetime import datetime
from typing import List

AMBIGUOUS = set("l1I|O0{}[]()/\\'\"`~,;.<>")
SYMBOLS = "!@#$%^&*()-_=+[]{};:,.<>?/|~"

def filtered_chars(chars: str, avoid_ambiguous: bool) -> List[str]:
    return [c for c in chars if c not in AMBIGUOUS] if avoid_ambiguous else list(chars)

def entropy_bits(length: int, pool_size: int) -> float:
    if length <= 0 or pool_size <= 1: return 0.0
    return length * math.log2(pool_size)

def categorize_strength(bits: float) -> str:
    if bits < 45: return "Weak"
    if bits < 75: return "Fair"
    if bits < 100: return "Good"
    return "Strong"

def make_password(length: int, use_lower: bool, use_upper: bool, use_digits: bool, use_symbols: bool, avoid_ambiguous: bool) -> str:
    categories = []
    if use_lower:  categories.append(filtered_chars(string.ascii_lowercase, avoid_ambiguous))
    if use_upper:  categories.append(filtered_chars(string.ascii_uppercase, avoid_ambiguous))
    if use_digits: categories.append(filtered_chars(string.digits, avoid_ambiguous))
    if use_symbols: categories.append(filtered_chars(SYMBOLS, avoid_ambiguous))
    if not categories: raise ValueError("Please select at least one character type.")
    pwd_chars = [secrets.choice(cat) for cat in categories]
    pool = [c for cat in categories for c in cat]
    if not pool: raise ValueError("No characters available after filtering; try unchecking 'avoid ambiguous'.")
    pwd_chars += [secrets.choice(pool) for _ in range(max(0, length - len(pwd_chars)))]
    # Fisher-Yates shuffle with secrets
    for i in range(len(pwd_chars)-1, 0, -1):
        j = secrets.randbelow(i+1)
        pwd_chars[i], pwd_chars[j] = pwd_chars[j], pwd_chars[i]
    return "".join(pwd_chars[:length])

class App(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=16)
        self.master.title("Password Generator")
        self.master.minsize(640, 420)
        style = ttk.Style(self)
        if "clam" in style.theme_names(): style.theme_use("clam")
        style.configure("Red.Horizontal.TProgressbar")
        style.configure("Yellow.Horizontal.TProgressbar")
        style.configure("Green.Horizontal.TProgressbar")

        self.var_length = tk.IntVar(value=16)
        self.var_lower = tk.BooleanVar(value=True)
        self.var_upper = tk.BooleanVar(value=True)
        self.var_digits = tk.BooleanVar(value=True)
        self.var_symbols = tk.BooleanVar(value=False)
        self.var_avoid_ambig = tk.BooleanVar(value=True)
        self.var_show = tk.BooleanVar(value=False)
        self.var_password = tk.StringVar(value="")

        self.history: List[str] = []

        self._build_layout()
        self.grid(row=0, column=0, sticky="nsew")
        self.master.rowconfigure(0, weight=1); self.master.columnconfigure(0, weight=1)

    def _build_layout(self):
        lf = ttk.LabelFrame(self, text="Options", padding=12)
        lf.grid(row=0, column=0, sticky="nsew", padx=(0,12))
        self.columnconfigure(0, weight=1); self.rowconfigure(1, weight=1)

        ttk.Label(lf, text="Length:").grid(row=0, column=0, sticky="w")
        self.spin_len = ttk.Spinbox(lf, from_=4, to=128, textvariable=self.var_length, width=8)
        self.spin_len.grid(row=0, column=1, sticky="w", padx=6, pady=4)

        ttk.Checkbutton(lf, text="Lowercase a-z", variable=self.var_lower).grid(row=1, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Checkbutton(lf, text="Uppercase A-Z", variable=self.var_upper).grid(row=2, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Checkbutton(lf, text="Digits 0-9", variable=self.var_digits).grid(row=3, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Checkbutton(lf, text="Symbols", variable=self.var_symbols).grid(row=4, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Checkbutton(lf, text="Avoid ambiguous (l,1,I,|,O,0...)", variable=self.var_avoid_ambig).grid(row=5, column=0, columnspan=2, sticky="w", pady=(2,8))

        btns = ttk.Frame(lf); btns.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(4,2))
        btns.columnconfigure(0, weight=1); btns.columnconfigure(1, weight=1)
        ttk.Button(btns, text="Generate", command=self.on_generate).grid(row=0, column=0, sticky="ew", padx=(0,4))
        ttk.Button(btns, text="Copy", command=self.on_copy).grid(row=0, column=1, sticky="ew", padx=(4,0))
        ttk.Button(lf, text="Save to file...", command=self.on_save).grid(row=7, column=0, columnspan=2, sticky="ew", pady=(2,0))

        rf = ttk.LabelFrame(self, text="Result", padding=12)
        rf.grid(row=0, column=1, rowspan=2, sticky="nsew"); self.columnconfigure(1, weight=2)

        entry_frame = ttk.Frame(rf); entry_frame.grid(row=0, column=0, sticky="ew"); entry_frame.columnconfigure(0, weight=1)
        self.ent_pwd = ttk.Entry(entry_frame, textvariable=self.var_password, show="•")
        self.ent_pwd.grid(row=0, column=0, sticky="ew")
        ttk.Checkbutton(entry_frame, text="Show", variable=self.var_show, command=self.toggle_show).grid(row=0, column=1, padx=8)

        meter_frame = ttk.Frame(rf); meter_frame.grid(row=1, column=0, sticky="ew", pady=(8,0)); meter_frame.columnconfigure(0, weight=1)
        ttk.Label(meter_frame, text="Strength:").grid(row=0, column=0, sticky="w")
        self.var_strength_label = tk.StringVar(value="—")
        ttk.Label(meter_frame, textvariable=self.var_strength_label).grid(row=0, column=1, sticky="w", padx=6)
        self.progress = ttk.Progressbar(meter_frame, orient="horizontal", mode="determinate", maximum=100, value=0, length=200)
        self.progress.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4,0))

        hist_frame = ttk.LabelFrame(rf, text="History (last 10)", padding=6)
        hist_frame.grid(row=2, column=0, sticky="nsew", pady=(10,0))
        rf.rowconfigure(2, weight=1); rf.columnconfigure(0, weight=1)
        self.listbox = tk.Listbox(hist_frame, height=6)
        self.listbox.grid(row=0, column=0, sticky="nsew"); hist_frame.rowconfigure(0, weight=1); hist_frame.columnconfigure(0, weight=1)
        sb = ttk.Scrollbar(hist_frame, orient="vertical", command=self.listbox.yview); self.listbox.configure(yscrollcommand=sb.set); sb.grid(row=0, column=1, sticky="ns")
        hist_btns = ttk.Frame(hist_frame); hist_btns.grid(row=1, column=0, sticky="ew", pady=(6,0))
        ttk.Button(hist_btns, text="Use Selected", command=self.on_use_selected).grid(row=0, column=0, sticky="w")
        ttk.Button(hist_btns, text="Clear History", command=self.on_clear_history).grid(row=0, column=1, sticky="w", padx=6)

        self.master.bind("<Control-g>", lambda e: self.on_generate())
        self.master.bind("<Control-c>", lambda e: self.on_copy())

    def toggle_show(self):
        self.ent_pwd.configure(show="" if self.var_show.get() else "•")

    def on_generate(self):
        try:
            length = int(self.var_length.get())
            if not (4 <= length <= 128): raise ValueError("Length must be between 4 and 128.")
            pwd = make_password(length, self.var_lower.get(), self.var_upper.get(), self.var_digits.get(), self.var_symbols.get(), self.var_avoid_ambig.get())
            self.var_password.set(pwd)
            self.update_strength(length)
            self.add_history(pwd)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_copy(self):
        pwd = self.var_password.get()
        if not pwd:
            messagebox.showinfo("Copy", "Generate a password first."); return
        try:
            self.master.clipboard_clear(); self.master.clipboard_append(pwd); self.master.update()
            messagebox.showinfo("Copy", "Password copied to clipboard.")
        except tk.TclError:
            messagebox.showwarning("Clipboard", "Could not access clipboard.")

    def on_save(self):
        pwd = self.var_password.get()
        if not pwd:
            messagebox.showinfo("Save", "Generate a password first."); return
        fn = filedialog.asksaveasfilename(title="Save password", defaultextension=".txt", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if fn:
            with open(fn, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {pwd}\n")
            messagebox.showinfo("Save", f"Saved to {fn}")

    def add_history(self, pwd: str):
        item = f"{datetime.now().strftime('%H:%M:%S')}  {pwd}"
        self.history.insert(0, item); self.history = self.history[:10]
        self.refresh_history_listbox()

    def refresh_history_listbox(self):
        self.listbox.delete(0, tk.END)
        for item in self.history: self.listbox.insert(tk.END, item)

    def on_use_selected(self):
        sel = self.listbox.curselection()
        if not sel: return
        item = self.history[sel[0]]
        parts = item.split("  ", 1)
        self.var_password.set(parts[1] if len(parts)>1 else "")
        self.update_strength(len(self.var_password.get()))

    def on_clear_history(self):
        self.history.clear(); self.refresh_history_listbox()

    def update_strength(self, length: int):
        pool = 0
        if self.var_lower.get():  pool += len(filtered_chars(string.ascii_lowercase, self.var_avoid_ambig.get()))
        if self.var_upper.get():  pool += len(filtered_chars(string.ascii_uppercase, self.var_avoid_ambig.get()))
        if self.var_digits.get(): pool += len(filtered_chars(string.digits, self.var_avoid_ambig.get()))
        if self.var_symbols.get(): pool += len(filtered_chars(SYMBOLS, self.var_avoid_ambig.get()))
        bits = entropy_bits(length, pool)
        label = f"{categorize_strength(bits)}  (~{bits:.1f} bits)"
        # simple mapping for bar fill
        percent = max(0, min(100, int(bits))) if bits < 100 else min(100, int((bits/128)*100))
        self.var_strength_label.set(label)
        style = "Red.Horizontal.TProgressbar" if bits < 45 else "Yellow.Horizontal.TProgressbar" if bits < 75 else "Green.Horizontal.TProgressbar"
        self.progress.configure(style=style, value=percent)

def main():
    root = tk.Tk()
    app = App(root)
    app.mainloop()

if __name__ == "__main__":
    main()

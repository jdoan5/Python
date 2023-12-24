import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

def save_file():
    file_path = filedialog.asksaveasfilename(defaultextension=".txt")
    if not file_path:
        return
    with open(file_path, "w") as file:
        text = text_area.get("1.0", tk.END)
        file.write(text)
    root.title(f"Simple Text Editor - {file_path}")

def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
    if not file_path:
        return
    with open(file_path, "r") as file:
        text = file.read()
        text_area.delete("1.0", tk.END)
        text_area.insert("1.0", text)
    root.title(f"Simple Text Editor - {file_path}")

def new_file():
    root.title("Simple Text Editor")
    text_area.delete("1.0", tk.END)

root = tk.Tk()
root.title("Simple Text Editor")

menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

file_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="New", command=new_file)
file_menu.add_command(label="Open...", command=open_file)
file_menu.add_command(label="Save As...", command=save_file)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.quit)

text_area = tk.Text(root)
text_area.pack(expand=True, fill=tk.BOTH)

root.mainloop()

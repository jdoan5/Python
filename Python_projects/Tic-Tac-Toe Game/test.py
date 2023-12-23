import tkinter as tk
from tkinter import messagebox

def show_message():
    messagebox.showinfo("Message", "Merry Christmas!")

def create_window():
    window = tk.Tk()
    window.title("Christmas App")
    window.geometry("300x200")  # Width x Height

    christmas_button = tk.Button(window, text="Click for Christmas Greeting", command=show_message)
    christmas_button.pack(pady=20)

    window.mainloop()

if __name__ == "__main__":
    create_window()

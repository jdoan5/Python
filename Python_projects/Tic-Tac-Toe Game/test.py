import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

def show_message():
    messagebox.showinfo("Message", "Merry Christmas!")
    display_image()

def display_image():
    image = Image.open("christmas.jpg")
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo

def create_window():
    global image_label
    window = tk.Tk()
    window.title("Christmas Greeting App")
    window.geometry("400x400")

    christmas_button = tk.Button(window, text = "Click Here")
    christmas_button.pack(pady=20)

    image_label = tk.Label(window)
    image_label.pack()

    window.mainloop()

if __name__ == "__main__":
    create_window()
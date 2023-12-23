import tkinter as tk
from PIL import Image, ImageTk

def on_image_click(event):
    print("Merry Christmas!")

# Create the main window
root = tk.Tk()
root.title("Merry Christmas App")

# Load an image
image = Image.open("christmas.jpn")
photo = ImageTk.PhotoImage(image)

# Add image to a label widget
label = tk.Label(root, image=photo)
label.pack()

# Bind click event
label.bind("<Button-1>", on_image_click)

# Start the GUI
root.mainloop()

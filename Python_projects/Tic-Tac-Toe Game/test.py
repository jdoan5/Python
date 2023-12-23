import tkinter as tk
from PIL import Image, ImageTk

def show_image():
    # Load an image
    image = Image.open("/Users/johndoan/Documents/GitHub/Python/image/panda.png")
    
    photo = ImageTk.PhotoImage(image)

    # Add image to a label widget and display
    image_label.config(image=photo)
    image_label.image = photo  # Keep a reference!
    TK_SILENCE_DEPRECATION=1 

# Create the main window
root = tk.Tk()
root.title("Clickable Text with Picture")

# Create a button widget
button = tk.Button(root, text="Click me!", command=show_image)
button.pack()

# Create an empty label for the image
image_label = tk.Label(root)
image_label.pack()

# Start the GUI
root.mainloop()

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from tkinter.font import Font
import random


class App:
    def __init__(self, root, image_path):
        self.root = root
        self.root.title("Christmas Random Messages App")

        # Load the background image
        self.bg_image = Image.open(image_path)
        self.bg_photo = ImageTk.PhotoImage(self.bg_image)

        # Set the image as background
        self.bg_label = tk.Label(root, image=self.bg_photo)
        self.bg_label.place(relwidth=1, relheight=1)  # Make label cover the whole window

        self.messages = ["Christmas is a time for remembering family and trying to guess everyone's sizes! Have a Wonderful Christmas!",
                          "I've finally found the true meaning of Xmas, it's for those people who can't spell Christmas!", 
                          "Christmas is mostly for children. But we adults can enjoy it too until the credit card bills arrive!", 
                          "Merry Christmas, ya filthy animal",]
        
        #set a window size
        self.root.geometry("900x600")

        #set font size for the "button"
        large_font = Font(family="Helvetica", size=15, weight="bold")

        

        #click on "button"
        self.button = tk.Button(root, text="Suprise Me", command=self.show_random_message,
                                 borderwidth=0, highlightthickness=0, font=large_font, relief='flat', bg='light blue')
        self.button.pack(expand=True)
 
    def show_random_message(self):
        random_message = random.choice(self.messages)
        messagebox.showinfo("Random Message", random_message)

def main():
    root = tk.Tk()
    root.geometry("800x600")  # Set this to the size of your image

    app = App(root, "image/christmas_2.jpg")  # Replace with your image path

    root.mainloop()

if __name__ == "__main__":
    main()

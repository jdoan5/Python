import tkinter as tk
from tkinter import messagebox
from tkinter.font import Font
from PIL import Image, ImageTk
import random


class App:
    def __init__(self, root,):
        self.root = root
        self.root.title("Random Christmas Messages")

        self.messages = ["Christmas is a time for remembering family and trying to guess everyone's sizes! Have a Wonderful Christmas!",
                          "I've finally found the true meaning of Xmas, it's for those people who can't spell Christmas!", 
                          "Christmas is mostly for children. But we adults can enjoy it too until the credit card bills arrive!", 
                          "Merry Christmas, ya filthy animal",]

        #set a window size
        self.root.geometry("400x300")

        #set a background color
        self.root.configure(bg="light coral")

        #set font size for the "button"
        large_font = Font(family="Helvetica", size=16, weight="bold")

        #click on "button"
        self.button = tk.Button(root, text="Suprise Me", command=self.show_random_message,
                                bg="light coral", fg="black", font=large_font)
        self.button.pack(expand=True)
        
        
        

    def show_random_message(self):
        random_message = random.choice(self.messages)
        messagebox.showinfo("Random Message", random_message)

def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()

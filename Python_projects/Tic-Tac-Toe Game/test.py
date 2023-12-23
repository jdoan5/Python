import tkinter as tk

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Click Message App")
        self.button = tk.Button(root, text="Click me!", command=self.on_click)
        self.button.pack()
        self.label = tk.Label(root, text="")
        self.label.pack()
        self.click_count = 0
        self.messages = ["First Click!", "Second Click!", "Third Click!", "More clicks!"]

    def on_click(self):
        message = self.messages[min(self.click_count, len(self.messages) - 1)]
        self.label.config(text=message)
        self.click_count += 1

# Create the main window
root = tk.Tk()
app = App(root)
root.mainloop()

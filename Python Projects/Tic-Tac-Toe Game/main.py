import tkinter as tk
from tkinter import messagebox

class TicTacToeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Full-Size Tic-Tac-Toe")
        self.initialize_ui()
        self.initialize_board()

    def initialize_ui(self):
        self.label = tk.Label(self.root, text="Player X's turn", font=('normal', 20))
        self.label.pack(side="top")
        self.restart_button = tk.Button(self.root, text="Restart Game", font=('normal', 15), command=self.reset_board)
        self.restart_button.pack(side="top")

    def initialize_board(self):
        self.board_frame = tk.Frame(self.root)
        self.board_frame.pack(expand=True, fill="both")
        self.buttons = [[None for _ in range(3)] for _ in range(3)]
        for row in range(3):
            self.board_frame.rowconfigure(row, weight=1)
            self.board_frame.columnconfigure(row, weight=1)
            for col in range(3):
                button = tk.Button(self.board_frame, text="", font=('normal', 40), height=2, width=4,
                                   command=lambda r=row, c=col: self.on_button_click(r, c))
                button.grid(row=row, column=col, sticky="nsew")
                self.buttons[row][col] = button

    def on_button_click(self, row, col):
        if self.buttons[row][col]['text'] == "" and not self.check_winner():
            self.buttons[row][col]['text'] = self.player_turn
            if self.check_winner():
                messagebox.showinfo("Game Over", f"Player {self.player_turn} wins!")
                self.reset_board()
            elif self.check_draw():
                messagebox.showinfo("Game Over", "It's a draw!")
                self.reset_board()
            else:
                self.player_turn = "O" if self.player_turn == "X" else "X"
                self.label['text'] = f"Player {self.player_turn}'s turn"

    def check_winner(self):
        # Check rows, columns and diagonals for a win
        for i in range(3):
            if self.buttons[i][0]['text'] == self.buttons[i][1]['text'] == self.buttons[i][2]['text'] != "":
                return True
            if self.buttons[0][i]['text'] == self.buttons[1][i]['text'] == self.buttons[2][i]['text'] != "":
                return True
        if self.buttons[0][0]['text'] == self.buttons[1][1]['text'] == self.buttons[2][2]['text'] != "":
            return True
        if self.buttons[0][2]['text'] == self.buttons[1][1]['text'] == self.buttons[2][0]['text'] != "":
            return True
        return False

    def check_draw(self):
        return all(all(button['text'] != "" for button in row) for row in self.buttons)

    def reset_board(self):
        for row in self.buttons:
            for button in row:
                button['text'] = ""
        self.player_turn = "X"
        self.label['text'] = "Player X's turn"

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("600x600")  # Set the window size to 600x600 pixels
    app = TicTacToeApp(root)
    root.mainloop()

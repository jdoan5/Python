# Initialize the board
board = [" " for _ in range(9)]

# Function to print the Tic Tac Toe board
def print_board(board):
    print(" " + board[0] + " | " + board[1] + " | " + board[2])
    print("-----------")
    print(" " + board[3] + " | " + board[4] + " | " + board[5])
    print("-----------")
    print(" " + board[6] + " | " + board[7] + " | " + board[8])

# Function to check for a win
def check_win(board, player):
    winning_combinations = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
    for combo in winning_combinations:
        if board[combo[0]] == board[combo[1]] == board[combo[2]] == player:
            return True
    return False

# Main game loop
current_player = "X"
while True:
    print_board(board)
    print(f"Player {current_player}'s turn. Enter a position (1-9): ")
    try:
        position = int(input()) - 1
        if 0 <= position <= 8 and board[position] == " ":
            board[position] = current_player
            if check_win(board, current_player):
                print_board(board)
                print(f"Player {current_player} wins! Congratulations!")
                break
            if " " not in board:
                print_board(board)
                print("It's a tie!")
                break
            current_player = "O" if current_player == "X" else "X"
        else:
            print("Invalid move. Please choose an empty position (1-9).")
    except ValueError:
        print("Invalid input. Please enter a number (1-9).")

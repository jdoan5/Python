import numpy as np

ROWS = 3
COLUMNS = 3

def mark(row, col, player):
    board[row][col] = player

def is_valid_mark(row,col):
    return board[row][col] == 0    

board = np.zeroes((ROWS,COLUMNS))

print(board)
mark(1,0,2)
print(board)
print(is_valid_mark(1,0))
# Example 1
matrix = [[k for k in range(3) for j in range(3)] for i in range(3)]
print(matrix[1][1][1])

# Example 2
matrix = [[[0, 1, 2], [0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2], [0, 1, 2]]]
print(matrix[0][0][0])

# Example 3
matrix = [[[k for k in range(3)] for j in range(3)] for i in range(3)]
print(matrix[0][0][1])

# Example 4
matrix = [[[0, 1, 2], [0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2], [0, 1, 2]]]
matrix2 = []
for submatrix in matrix:
  for val in submatrix:
    matrix2.append(val)
print(matrix2[2][2])

# Example 5
# Choose the correct answer to get the “Red” color from the following list:
Colors= [ [['Blue','Green','White','Black']], [['Green','Blue','White','Yellow']] , [['White','Blue','Red','Green']] ]

# Example 6
matrix = [[[k for k in range(3)] for j in range(3)] for i in range(3)]
print(matrix[2][1])

# Exaple 7
matrix = [[[0, 1, 2], [0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2], [0, 1, 2]]]
matrix2 = []
for submatrix in matrix:
  for val in submatrix:
    matrix2.append(val)
print(matrix2[2])

# Example 8
matrix = [[[k for k in range(3)] for j in range(3)] for i in range(3)]
print(matrix[1][2])


# Example 9
matrix = [[[0, 1, 2], [0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2], [0, 1, 2]]]
matrix2 = []
for submatrix in matrix:
  for val in submatrix:
    matrix2.append(val)
print(matrix2[2][0])

# Example 10
# Choose the correct code to get the third element in the second row, Regarding the following list :
Colors = [ ['Red', 'Green', 'White', 'Black'], ['Green', 'Blue', 'White', 'Yellow'] ,['White', 'Blue', 'Green', 'Red'] ]

# Example 1
# This is an example to create a 2-dimensional list 'a' with dimensions 5x5 
a = []
for i in range(5):
    a.append([])
    for j in range(5):
        a[i].append(j)
print(a[3][3])


# Example 2
# Here is a 2-dimensional list "matrix" with dimensions 3x3 where each element is initialized with the value of 'j' from the inner loop
# Then, it prints the value at index [1][2] 
matrix = [[j for j in range(3)] for i in range(3)] 
print(matrix[1][2])

# Example 3
matrix = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
matrix2 = []
for submatrix in matrix:
    for val in submatrix:
        matrix2.append(val)
print(matrix2[0])

# Example 4
countries = [['Egypt', 'USA', 'India'], ['Dubai', 'America,' 'Spain'], ['London', 'England', 'France']]
countries2 = [country for sublist in countries for country in sublist if len(country) < 6]
print(countries2)

# Example 5 
a = []
for i in range(5):
    a.append([])
    for j in range(5):
        a[i].append(j)
print(a[2][3])

# Example 6
countries = [['Egypt', 'USA', 'India'], ['Dubai', 'America', 'Spain'], ['London', 'England', 'France']]
countries2 = [country for sublist in countries for country in sublist if len(country) < 4]
print(countries2)

# Example 7
matrix = [[j for j in range(4)] for i in range(4)]
print(matrix[3][1])

# Example 8
matrix = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
matrix2 = []
for submatrix in matrix:
  for val in submatrix:
    matrix2.append(val)
print(matrix2[2])

# Example 9
matrix = [[j for j in range(3)] for i in range(3)] 
print(matrix[2][1])

# Example 10
a = []
for i in range(2):
    a.append([])
    for j in range(2):
        a[i].append(j)
print(a)


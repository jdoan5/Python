# Example 1
# This is an example to create a 2-dimensional list 'a' with dimensions 5x5 
a = []
for i in range(5):
    a.append([])
    for j in range(5):
        a[i].append(j)
print(a[3][3])


# Example 2
matrix = [[j for j in range(3)]]
# Example 1
# What will be the output of below Python code?
# Here enumrate is a function that allows you to keep track of the number iterations (loops) in a loop. 
# Enumerate object cotains a count (always start count from 0)
list1 = [1, 2, 3, 4, 5]
for index, j in enumerate(list1):
    print(index,j)

# Example 2
# What will be the output of below Python code?
# In this code, the outler loop iterates over the sublists of 'list1'. If the sublist's length is 4 then it is printed
list1 = [[1,2,3,5], [4,5,6,7],[8,9,10],[8,1,3,3,1]]
for i in list1:
    if len(i)==4:
        print(i)

# Example 3
# What will be the output of below Python code?
# Lists are zero-indexed, therefore the position of first element at 0 is 10
list1 = [10, 11, 12, 13, 14]
print(list1[0])

# Example 4
list1 = [10, 11, 12, 13, 14]*3
print(list1)

# Example 5
# What will be the output of below Python code?
# Here, the 'append()' means to add a single item to the existing list
list1 = [10, 11, 12, 13, 14]
list1.append(15)
print(list1)

# Example 6
# Here, [::-1] means to start at the end of the string and end at position 0
# Next, moving with step '-1' which means one step backwards 
list1 = [4, 1, 3, 2, 6]
print(list1[::-1])

# Example 7
# Here, [::-1] means to start at the end of the string and end at position 0
# Next, moving with step '1' which means one step forwards 
list1 = [10, 15, 12, 19, 18]
print(list1[::1])

# Example 8
# Here, list slicing provides a way to access elements of the list
# The syntax of list slicing as 'start:stop:step'
# Here, index starts from 1 (the second item)
letters = ["B", "K", "F", "G", "M"]
print(letters[1:])

# Example 9
# Here, lists are zero-indexed. The index at 0, then the position is 10
list1 = [10, 15, 19, 29, 14]
print(list1[0])
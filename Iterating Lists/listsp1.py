# Example 1
# The purpose of this code is to print the sum of all numbers in the "values" list 
# Initialize sum then add  "number" to "sum"
# The loop finishes, "sum" will print out the total numbers in "values"
# the result = 10
sum = 0 
values = [0, 1, 2, 3, 4]
for number in values:
    sum = sum + number
print(sum)


# Example 2
# Same method above but line 18, condense the sum in a short cut
sum = 0
values = [2, 9, 1, 7]
for number in values:
    sum += number
print(sum)    


# Example 3
# how many times a loop runs ?
# In this loop, Python will iterate over each item in the list
# There are 5 itmes, so the loop will run 5 times
for i in [9, 1, 5, 3, 2]:
    print(i)


# Example 4
# Which of the following statements won’t be printed when this Python code is run?
# The string will not print "o" caused the loop to immediately move to the next iteration
for letter in "Working":
    if letter == 'o':
        continue
    print('Letter : ' + letter )


# Example 5
# How many asterisks will be printed when the following code executes?
# In this nested loop, for each value of x and the inner loop of y, it will print a "#"
# The outer loop runs 5 times because there are 5 elements and the inter loop runs 4 times because there are 4 elements
# Thefore, 5 * 4 = 20. Print out '#' 20 times
for x in [1, 3, 4, 2, 2]:
    for y in [2, 3, 2, 1]:
        print('#')
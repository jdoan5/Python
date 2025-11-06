# Write a program that adds the digits in a 2 digit numbers
# ex: if the input is 32, then the output should be 3 + 2 = 5

user_input = input("Enter a two digit numbers: ")

# convert string into int
n1 = int(user_input[0])
n2 = int(user_input[1])

# adding two numbers
result = n1 + n2
print(result)


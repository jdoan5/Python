# A program to calculate the Body Mass Index (BMI)

# User input
weight = int(input("Enter your weight in kilograms: "))
height = float(input("Enter your height in meters: "))

# Calculated the input
result = weight / (height ** 2)

# Convert the result into integer instead a float result 
print(int(result))
# Example 1
# Regarding the definition of the function and the sample input, Choose the correct value of the output.
#Function :  def multi_func(num1,num2):
#                      return num1 *num2 
# Sample input: print ( multi_func(5 , num1= 10) )

# Example 2
def my_function(*students):
  print("The tallest student is " + students[2])
my_function("James", "Ella", "Jackson")

# Example 3
# What's the error in the following snippet code
def multi_func():
  result = int(input()) * 5
  return result     
print(result)
# Example 1
# Here is an example of slicing syntax. In this example, my_list with a step size of 3 then it skips every two elements and only includes every third element
# The result contains a new list [0, 3]
my_list = [0, 1, 2, 3, 4]
print(my_list[::3])

# Example 2
# Here is an example of list slicing. In this example, it selects a portion of the list 'my_list' starting from the index 2, and up too but not including the index 4
# The result contains [2, 3]
my_list = [0, 1, 2, 3, 4]
print(my_list[2:4])

# Example 3
# Here is an example of slicing syntax. In this example, my_list with a step size of 3 then it skips every two elements and only includes every second element
# The result contains a new list [0, 2, 4]
my_list = [0, 1, 2 , 3, 4]
print(my_list[::2])

# Example 4
# Here is example of print a slice. In this example, my_list starting from the element at index 2 until end of the list along a new element.
my_list = [0, 1, 2, 3, 4]
my_list.append('hi')
print(my_list[2:])

# Example 5
# Here is an example of reverse order. In this example, my_list will create a new list from the last element (index - 1) and going backward by a step of -1
my_list = [0, 1, 2, 3, 4]
print(my_list[::-1])

# Example 6
# Here is an example print a slice of 'lists1'. In this example, lists1 returns a new list containing the elements at indcies 0,1,2,3
# Here, the element [11,55,"cat"] is nested at index 3
list1= [1, 66, "Python", [11, 55, "cat"], [], 2.22, True]
print(list1[0:4])

# Example 7
# Another example of reserve order. In this example, it will be accessed from the element at index -1, which is the last element in the list
my_list = [0, 1, 2, 3, 4]
print(my_list[-1])


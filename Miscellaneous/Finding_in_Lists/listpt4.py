# Example 1
# Here, (4,6) is not in the list and there are three elements here
(4, 6) not in [(4, 7), (5, 6), "hello"]

# Example 2
# In this example, the index() method is used to find the index of the first occurence of a specified element in a list.
# Here, the result is 3 returns at index(1)
my_list = [0, 3, 4, 1, 2]
print(my_list.index(1))

# Example 3
# Here is an example to modifies the sublist from index 3 (inclusive) to index 5 (exclusive) and replaces it with the elements [10,11]
list1 = [1, 3, 5, 9, 0, 2, 6, 7]
list1[3:5] = [10, 11]
print(list1)

# Exampl 4
# Here, it will replace index 1 with the element [8, 10]
list1 = [0, 3, 4, 1, 2]
list1[1] = [8, 10]
print(list1)

# Example 5
list1 = [3, 2, 1, 5, 6]
list2 = list1
list1[1] = 9
print(list2)

# Example 6
# In this example, it swaps the positions of elements at index 0 and index 2. After executing the lists then values at index 0 and index 1 are swapped.
countries = ["USA", "CANADA", "CHINA"]
countries[0], countries[2] = countries[2], countries[0]
print(countries)

# Example 7
# Here is another example of modifies the sublist of 'list1' from index 1(inclusive) to index 5 (exclusive)
list1 = [0, 3, 1, 4, 5, 2]
list1[1:5]=[1,3]
print(list1)


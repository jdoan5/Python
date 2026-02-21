#Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
#You may assume that each input would have exactly one solution, and you may not use the same element twice.
#You can return the answer in any order.
from typing import List

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        seen = {} # values -> index
        for i, num in enumerate(nums):
            need = target - num
            if need in seen:
                return [seen[need], i]
            seen[num] = i
        return []

#Test Cases
s = Solution()
print(s.twoSum([2,7,11,15], 9))
print(s.twoSum([3,2,4], 6))
print(s.twoSum([3,3], 6))
print(s.twoSum([3,2,3], 6))
print(s.twoSum([3,3,3], 6))

# Explanation: The twoSum method takes a list of integers 'nums' and a target integer 'target'.
# It uses a dictionary 'seen' to store the indices of the numbers encountered so far.
# For each number in 'nums', it calculates the 'need' which is the difference between the target and the current number.
# If 'need' is found in 'seen', it returns the indices of 'need' and the current number.
# If 'need' is not found, it adds the current number and its index to 'seen'. If no solution is found, it returns an empty list.
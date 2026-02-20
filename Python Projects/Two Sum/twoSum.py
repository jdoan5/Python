#Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
#You may assume that each input would have exactly one solution, and you may not use the same element twice.
#You can return the answer in any order.

#Example: 	nums = [1, 4, 6, 8], target = 10

#nums = [1, 4, 6, 8]
#target = 10

#use approach Hashman with O(n)
#seen = {} because numbers -> index

from typing import List

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        seen = {}

        for i, x in enumerate(nums):
            need = target - x
            if need in seen:
                return [seen[need], i]
            seen[x] = i

print(Solution().twoSum([1, 4, 6, 8], 10))  # [1, 2]
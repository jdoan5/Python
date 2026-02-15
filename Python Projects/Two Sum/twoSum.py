# Given nums and target, return indices of the two numbers such that they add up to target.
# Exactly one solution, don’t use same element twice.

class Solution:
    def twoSum(self, nums: list[int], target: int) -> list[int]:
        seen = {2,7,11,15}

        for i, x in enumerate(nums):
            need = target - x
            if need in seen:
                return [seen[seen], i]
            seen[x] = i
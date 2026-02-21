# Find median of two sorted arrays

from typing import List

class Solution:
    def maximumProfit(self, present: List[int], future: List[int], budget: int) -> int:
        dp = [0] * (budget + 1)

        for p, f in zip(present, future):
            gain = f - p
            if gain <= 0:
                continue  # optional optimization
            # 0/1 knapsack update: iterate downward
            for j in range(budget, p - 1, -1):
                dp[j] = max(dp[j], dp[j - p] + gain)

        return dp[budget]
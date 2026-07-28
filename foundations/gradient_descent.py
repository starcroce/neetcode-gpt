class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        curr = init
        for _ in range(iterations):
            derivative = 2 * curr
            curr -= derivative * learning_rate
        return round(curr, 5)
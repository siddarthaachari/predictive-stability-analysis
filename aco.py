import numpy as np


class ACO:

    def __init__(self, iterations=20, ants=10):

        self.iterations = iterations
        self.ants = ants
        self.history = []

    def solve(self, fitness_func):

        best_solution = None
        best_score = float("inf")

        for i in range(self.iterations):

            for j in range(self.ants):

                solution = np.random.uniform(0.001, 10, 1)

                score = fitness_func(solution)

                if score < best_score:
                    best_score = score
                    best_solution = solution

            self.history.append(best_score)

        return best_solution, best_score

import numpy as np
import random
from typing import List, Tuple, Optional
import sys

class GeneticAlgorithm:
    def __init__(
        self,
        pop_size:           int,            # Population size
        generations:        int,            # Number of generations for the algorithm
        mutation_rate:      float,          # Gene mutation rate
        crossover_rate:     float,          # Gene crossover rate
        tournament_size:    int,            # Tournament size for selection
        elitism:            bool,           # Whether to apply elitism strategy
        random_seed:        Optional[int]   # Random seed for reproducibility
    ):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism = elitism

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def _init_population(self, M: int, N: int) -> List[List[int]]:
        population = []
        for _ in range(self.pop_size):
            individual = self._generate_valid_individual(M, N)
            population.append(individual)
        return population

    def _generate_valid_individual(self, M: int, N: int) -> List[int]:
        individual = [random.randint(0, M - 1) for _ in range(N)]
        while len(set(individual)) < M:  # Ensure every technician gets at least one task
            unassigned_technicians = set(range(M)) - set(individual)
            for tech in unassigned_technicians:
                random_task = random.randint(0, N - 1)
                individual[random_task] = tech
        return individual

    def _fitness(self, individual: List[int], technician_times: np.ndarray) -> float:
        total_time = 0
        penalty = 1e6  # A large penalty for infeasible assignments
    
        for task, technician in enumerate(individual):
            task_time = technician_times[technician, task]
            if task_time == sys.maxsize:
                total_time += penalty  # Add penalty for infeasible task assignment
            else:
                total_time += task_time
    
        return total_time


    def _selection(self, population: List[List[int]], fitness_scores: List[float]) -> List[int]:
        tournament = random.sample(list(zip(population, fitness_scores)), self.tournament_size)
        tournament.sort(key=lambda x: x[1])  # Sort by fitness score (lower is better)
        return tournament[0][0]  # Return the individual with the best fitness score

    def _crossover(self, parent1: List[int], parent2: List[int], M: int) -> Tuple[List[int], List[int]]:
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, len(parent1) - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
        else:
            child1, child2 = parent1[:], parent2[:]  # No crossover, just copy parents

        # Ensure validity (each technician gets at least one task)
        return self._ensure_valid(child1, M), self._ensure_valid(child2, M)

    def _mutate(self, individual: List[int], M: int) -> List[int]:
        if random.random() < self.mutation_rate:
            mutation_point = random.randint(0, len(individual) - 1)
            individual[mutation_point] = random.randint(0, M - 1)  # Assign task to a random technician
        return self._ensure_valid(individual, M)

    def _ensure_valid(self, individual: List[int], M: int) -> List[int]:
        while len(set(individual)) < M:
            unassigned_technicians = set(range(M)) - set(individual)
            for tech in unassigned_technicians:
                random_task = random.randint(0, len(individual) - 1)
                individual[random_task] = tech
        return individual

    def __call__(self, M: int, N: int, technician_times: np.ndarray) -> Tuple[List[int], int]:
        # Initialize population
        population = self._init_population(M, N)
        
        for _ in range(self.generations):
            # Calculate fitness for all individuals
            fitness_scores = [self._fitness(individual, technician_times) for individual in population]
            
            # Elitism: retain the best individual
            if self.elitism:
                best_individual = min(zip(population, fitness_scores), key=lambda x: x[1])[0]
            
            # Selection and reproduction
            new_population = []
            for _ in range(self.pop_size // 2):
                parent1 = self._selection(population, fitness_scores)
                parent2 = self._selection(population, fitness_scores)
                child1, child2 = self._crossover(parent1, parent2, M)
                new_population.append(self._mutate(child1, M))
                new_population.append(self._mutate(child2, M))
            
            if self.elitism:
                # Replace the worst individual with the best from the previous generation
                worst_index = fitness_scores.index(max(fitness_scores))
                new_population[worst_index] = best_individual
            
            population = new_population

        # Final fitness evaluation
        fitness_scores = [self._fitness(individual, technician_times) for individual in population]
        best_individual = min(zip(population, fitness_scores), key=lambda x: x[1])
        return best_individual[0], int(best_individual[1])

if __name__ == "__main__":

    def write_output_to_file(problem_num: int, total_time: int, filename: str = "answer.txt") -> None:

        print(f"Problem {problem_num}: Total time = {total_time}")

        if not isinstance(total_time, int):
            raise ValueError(f"Invalid format for problem {problem_num}. Total time should be an integer.")

    # Append result to file
        with open(filename, 'a') as file:
            file.write(f"Total time = {total_time}\n")


    M1, N1 = 2, 3
    cost1 = [[3, 2, 4],
             [4, 3, 2]]
    
    M2, N2 = 4, 4
    cost2 = [[5, 6, 7, 4],
             [4, 5, 6, 3],
             [6, 4, 5, 2],
             [3, 2, 4, 5]]
    
    M3, N3 = 8, 9
    cost3 = [[90, 100, 60, 5, 50, 1, 100, 80, 70],
             [100, 5, 90, 100, 50, 70, 60, 90, 100],
             [50, 1, 100, 70, 90, 60, 80, 100, 4],
             [60, 100, 1, 80, 70, 90, 100, 50, 100],
             [70, 90, 50, 100, 100, 4, 1, 60, 80],
             [100, 60, 100, 90, 80, 5, 70, 100, 50],
             [100, 4, 80, 100, 90, 70, 50, 1, 60],
             [1, 90, 100, 50, 60, 80, 100, 70, 5]]
    
    M4, N4 = 3, 3
    cost4 = [[2, 5, 6],
             [4, 3, 5],
             [5, 6, 2]]
    
    M5, N5 = 4, 4
    cost5 = [[6, 1, 5, 4],
             [6, 2, 1, 9],
             [5, 3, 9, 6],
             [2, 5, 4, 2]]
    
    M6, N6 = 4, 4
    cost6 = [[5, 4, 6, 7],
             [8, 3, 4, 6],
             [6, 7, 3, 8],
             [7, 8, 9, 2]]
    
    M7, N7 = 4, 4
    cost7 = [[4, 7, 8, 9],
             [6, 3, 6, 7],
             [8, 6, 2, 6],
             [7, 8, 7, 3]]
    
    M8, N8 = 5, 5
    cost8 = [[8, 8, 24, 24, 24],
             [6, 18, 6, 18, 18],
             [30, 10, 30, 10, 30],
             [21, 21, 21, 7, 7],
             [27, 27, 9, 27, 9]]
    
    M9, N9 = 5, 5
    cost9 = [[10, 10, sys.maxsize, sys.maxsize,sys.maxsize],
             [12, sys.maxsize, sys.maxsize, 12,12],
             [sys.maxsize, 15, 15, sys.maxsize,sys.maxsize],
             [11, sys.maxsize, 11, sys.maxsize,sys.maxsize],
             [sys.maxsize, 14, sys.maxsize, 14,14]]
    
    M10, N10 = 9, 10
    cost10 = [[1, 90, 100, 50, 70, 20, 100, 60, 80, 90],
              [100, 10, 1, 100, 60, 80, 70, 100, 50, 90],
              [90, 50, 70, 1, 100, 100, 60, 90, 80, 100],
              [70, 100, 90, 5, 10, 60, 100, 80, 90, 50],
              [50, 100, 100, 90, 20, 4, 80, 70, 60, 100],
              [100, 5, 80, 70, 90, 100, 4, 50, 1, 60],
              [90, 60, 50, 4, 100, 90, 100, 5, 10, 80],
              [100, 70, 90, 100, 4, 60, 1, 90, 100, 5],
              [80, 100, 5, 60, 50, 90, 70, 100, 4, 1]]

    problems = [(M1, N1, np.array(cost1)),
                (M2, N2, np.array(cost2)),
                (M3, N3, np.array(cost3)),
                (M4, N4, np.array(cost4)),
                (M5, N5, np.array(cost5)),
                (M6, N6, np.array(cost6)),
                (M7, N7, np.array(cost7)),
                (M8, N8, np.array(cost8)),
                (M9, N9, np.array(cost9)),
                (M10, N10, np.array(cost10))]

    # Example for GA execution:
    ga = GeneticAlgorithm(
        pop_size=100,
        generations=500,
        mutation_rate=0.1,
        crossover_rate=0.8,
        tournament_size=5,
        elitism=True,
        random_seed=59
    )

    # Solve each problem and immediately write the results to the file
    for i, (M, N, technician_times) in enumerate(problems, 1):
        best_allocation, total_time = ga(M=M, N=N, technician_times=technician_times)
        write_output_to_file(i, total_time)

    print("Results have been written to answer.txt")

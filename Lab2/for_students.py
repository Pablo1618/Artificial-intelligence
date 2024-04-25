from itertools import compress
import random
import time
import matplotlib.pyplot as plt
import copy

from data import *

def initial_population(individual_size, population_size):
    return [[random.choice([True, False]) for _ in range(individual_size)] for _ in range(population_size)]

def fitness(items, knapsack_max_capacity, individual):
    total_weight = sum(compress(items['Weight'], individual))
    if total_weight > knapsack_max_capacity:
        return 0
    return sum(compress(items['Value'], individual))

def population_best(items, knapsack_max_capacity, population):
    best_individual = None
    best_individual_fitness = -1
    for individual in population:
        individual_fitness = fitness(items, knapsack_max_capacity, individual)
        if individual_fitness > best_individual_fitness:
            best_individual = individual
            best_individual_fitness = individual_fitness
    return best_individual, best_individual_fitness

def crossover(parent1, parent2):
    crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(solution, mutation_rate):
    for i in range(len(solution)):
        if random.random() < mutation_rate:
            solution[i] = not solution[i]
    return solution

def elitism(population, items, knapsack_max_capacity, fitness_values, n_elite):
    #fitness_values = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    sorted_population = [x for _, x in sorted(zip(fitness_values, population), reverse=True)]
    elite = sorted_population[:n_elite]
    return elite

def select_parents_roulette(population, fitness_values, sum_fitness_values, n_selection):
    chancesToSelect = [fitness / sum_fitness_values for fitness in fitness_values]
    parents = random.choices(population, weights=chancesToSelect, k=n_selection)
    return parents


items, knapsack_max_capacity = get_big()
print(items)

population_size = 100
generations = 200
n_selection = 20
n_elite = 5
mutation_rate = 0.03

start_time = time.time()
best_solution = None
best_fitness = 0
population_history = []
best_history = []
population = initial_population(len(items), population_size)
#print(population[1])


for _ in range(generations):
    population_history.append(copy.deepcopy(population))
    children_population = []

    fitness_values = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    sum_fitness_values = sum(fitness_values)
    # Wybor zbioru rodzicow (selekcja ruletki)
    parents = select_parents_roulette(population, fitness_values, sum_fitness_values, n_selection)

    # Tworzenie nowego pokolenia osobnikow (rozwiazan)
    for _ in range(population_size // 2 - n_elite):
        # Wybor rodzicow z wyselekcjonowanego zbioru
        parent1, parent2 = random.sample(parents, k=2)
        child1, child2 = crossover(parent1, parent2)
        child1 = mutate(child1, mutation_rate)
        child2 = mutate(child2, mutation_rate)
        children_population.extend([child1, child2])

    # Elityzm
    elite = elitism(population, items, knapsack_max_capacity, fitness_values, n_elite)
    elite = [mutate(individual, mutation_rate) for individual in elite]
    children_population.extend(elite)

    # Metoda pokoleniowa - zastepujemy populacje
    population = children_population

    # Aktualizacja najlepszego rozwiazania ze zbioru populacji (rozwiazan)
    for individual in population:
        fitness_value = fitness(items, knapsack_max_capacity, individual)
        if fitness_value > best_fitness:
            best_solution = individual
            #print(best_solution)
            best_fitness = fitness_value
    best_history.append(best_fitness)


end_time = time.time()
total_time = end_time - start_time
total_weight = sum(compress(items['Weight'], best_solution))
print("Total weight: " + str(total_weight))
print("Max knapsack weight:" + str(knapsack_max_capacity))
print('Best solution:', list(compress(items['Name'], best_solution)))
print('Best solution value:', best_fitness)
print('Time: ', total_time)

# plot generations
x = []
y = []
top_best = 10
for i, population in enumerate(population_history):
    plotted_individuals = min(len(population), top_best)
    x.extend([i] * plotted_individuals)
    population_fitnesses = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    population_fitnesses.sort(reverse=True)
    y.extend(population_fitnesses[:plotted_individuals])
plt.scatter(x, y, marker='.')
plt.plot(best_history, 'r')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()

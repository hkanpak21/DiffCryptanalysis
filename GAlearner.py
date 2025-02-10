import numpy as np
import random

# ------------------------------------------------------------------
# 🎯 **Tunable Parameters for Genetic Algorithm**
# ------------------------------------------------------------------
POPULATION_SIZE = 100    # Number of candidates per generation
MUTATION_RATE = 0.05     # Probability of flipping a bit
GENERATIONS = 10000        # Number of generations
TOURNAMENT_SIZE = 5      # Tournament selection size
ELITISM = True           # Preserve best individuals each generation
CROSSOVER_RATE = 0.7     # Probability of crossover

# ------------------------------------------------------------------
# 1) Define DES S-box (S-box 1)
# ------------------------------------------------------------------
S_BOX = [
    [14,  4, 13,  1,  2, 15, 11,  8,  3, 10,  6, 12,  5,  9,  0,  7],
    [ 0, 15,  7,  4, 14,  2, 13,  1, 10,  6, 12, 11,  9,  5,  3,  8],
    [ 4,  1, 14,  8, 13,  6,  2, 11, 15, 12,  9,  7,  3, 10,  5,  0],
    [15, 12,  8,  2,  4,  9,  1,  7,  5, 11,  3, 14, 10,  0,  6, 13]
]

def apply_sbox(x):
    row = ((x & 0b100000) >> 4) | (x & 0b000001)
    col = (x & 0b011110) >> 1
    return S_BOX[row][col]

# ------------------------------------------------------------------
# 2) Helper Functions for Genetic Algorithm
# ------------------------------------------------------------------

def encode(A, B):
    """Encodes A (6x4 matrix) and B (6-bit vector) into a 30-bit integer."""
    A_bits = sum((A[j] << (6 * j)) for j in range(4))  # Flatten matrix into 24 bits
    return (A_bits << 6) | B  # Append B to the last 6 bits

def decode(individual):
    """Decodes a 30-bit integer back into A (6x4 matrix) and B (6-bit vector)."""
    B = individual & 0x3F  # Last 6 bits
    A_bits = (individual >> 6) & 0xFFFFFF  # First 24 bits
    A = [(A_bits >> (6 * j)) & 0x3F for j in range(4)]  # Extract columns
    return A, B

def fitness(individual):
    """Evaluates how well (A, B) approximates the S-box."""
    A, B = decode(individual)
    mismatches = 0
    for x in range(64):
        real_val = apply_sbox(x)
        x_prime = x ^ B
        # Approximate output
        approx_bits = sum(((bin(A[j] & x_prime).count('1') % 2) << j) for j in range(4))
        if approx_bits != real_val:
            mismatches += 1
    return -mismatches  # Negative because GA maximizes fitness

def mutate(individual):
    """Mutates a 30-bit integer by flipping random bits."""
    mask = sum((1 << i) for i in range(30) if random.random() < MUTATION_RATE)
    return individual ^ mask  # XOR with mutation mask

def crossover(parent1, parent2):
    """Performs single-point crossover between two 30-bit individuals."""
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, 29)  # Avoid 0-bit crossover
        mask = (1 << point) - 1  # Generate a mask like 000...1111 (point bits set)
        child1 = (parent1 & mask) | (parent2 & ~mask)
        child2 = (parent2 & mask) | (parent1 & ~mask)
        return child1, child2
    return parent1, parent2

def select(population, fitness_scores):
    """Tournament selection: pick the best of random competitors."""
    tournament = random.sample(list(zip(population, fitness_scores)), TOURNAMENT_SIZE)
    return max(tournament, key=lambda x: x[1])[0]  # Return best individual

# ------------------------------------------------------------------
# 3) Main Evolutionary Loop
# ------------------------------------------------------------------

def genetic_algorithm():
    """Runs the genetic algorithm to evolve (A, B)."""
    # Initialize random population
    population = [random.getrandbits(30) for _ in range(POPULATION_SIZE)]
    best_individual = None
    best_score = float('-inf')

    for gen in range(GENERATIONS):
        # Evaluate fitness
        fitness_scores = [fitness(ind) for ind in population]

        # Track best solution
        max_fitness = max(fitness_scores)
        if max_fitness > best_score:
            best_score = max_fitness
            best_individual = population[fitness_scores.index(max_fitness)]
        
        print(f"Gen {gen}: Best mismatch = {-best_score}")

        # Selection & Reproduction
        new_population = []
        if ELITISM:
            new_population.append(best_individual)  # Preserve best

        while len(new_population) < POPULATION_SIZE:
            parent1 = select(population, fitness_scores)
            parent2 = select(population, fitness_scores)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1), mutate(child2)])

        population = new_population[:POPULATION_SIZE]  # Ensure size consistency

    # Decode final best result
    best_A, best_B = decode(best_individual)
    return best_A, best_B, -best_score

# ------------------------------------------------------------------
# 4) Run the Genetic Algorithm
# ------------------------------------------------------------------

best_A, best_B, best_mismatch = genetic_algorithm()

# ------------------------------------------------------------------
# 5) Output the Best Approximation Found
# ------------------------------------------------------------------

print("\nBest Approximation Found:")
print(f"  B = {best_B:06b} (decimal: {best_B})")
print("  A (6x4 binary matrix, column-wise):")
for j in range(4):
    print(f"    Column {j}: {best_A[j]:06b} (decimal: {best_A[j]})")
print(f"\n  Best mismatch count = {best_mismatch} out of 64.")

# Final check: Compare real vs. approx outputs
print("\nFinal Comparison:")
print("+--------+--------------+-----------------+")
print("|  x(6)  | S-Box Output | Approx Output   |")
print("+--------+--------------+-----------------+")
for x in range(64):
    real_val = apply_sbox(x)
    x_prime = x ^ best_B
    approx_val = sum(((bin(best_A[j] & x_prime).count('1') % 2) << j) for j in range(4))
    print(f"| {x:06b} |   {real_val:04b} ({real_val:2d})  |   {approx_val:04b} ({approx_val:2d})   |")
print("+--------+--------------+-----------------+")

import numpy as np
import pandas as pd
import random
import math
import time

# --- Load Lookup Table and Fitness Data ---
DATASET_NAME = "Heart_Disease_Statlog" # Or Heart_Disease_Statlog, Seeds
MODEL_NAME = "RandomForestClassifier"
LOOKUP_TABLE_FILENAME = f"tables/lookup_table_{DATASET_NAME}_{MODEL_NAME}.csv"

print(f"Loading lookup table for BPSO: {LOOKUP_TABLE_FILENAME}")
n_features = 0
try:
    df = pd.read_csv(LOOKUP_TABLE_FILENAME)
    if 'subset_bitmask' not in df.columns or 'num_features' not in df.columns or 'fitness_h' not in df.columns:
         raise ValueError("Required columns missing in lookup table.")

    df['subset_bitmask'] = df['subset_bitmask'].astype(str)
    n_features = int(df['num_features'].max())
    df['subset_bitmask'] = df['subset_bitmask'].str.zfill(n_features)
    df.dropna(subset=['fitness_h'], inplace=True)
    if df.empty:
        raise ValueError("No valid data after removing NaNs.")

    fitness_dict = pd.Series(df.fitness_h.values, index=df.subset_bitmask).to_dict()
    print(f"Lookup table loaded for {DATASET_NAME}. N={n_features}. Number of solutions: {len(fitness_dict)}")

except Exception as e:
    print(f"ERROR during loading lookup table: {e}")
    exit()

# --- BPSO Parameters ---
SWARM_SIZE = 30      # Number of particles in swarm
# N_FEATURES = n_features (defined above)
N_ITERATIONS = 100   # Number of iterations (generations)
W_MAX = 0.9          # Start value for inertia weight
W_MIN = 0.4          # End value for inertia weight
C1 = 2.0             # Cognitive constant (attraction to pbest)
C2 = 2.0             # Social constant (attraction to gbest)
V_MAX = 4.0          # Maximum absolute value for velocity

# --- Helper functions ---

def array_to_bitstring(individual_array):
    """Converts numpy array [0, 1, 0] to bitstring '010'."""
    return "".join(individual_array.astype(str))

def get_fitness(individual_array, fitness_lookup):
    """Retrieves fitness from lookup dictionary. Returns infinity if not found."""
    bitstring = array_to_bitstring(individual_array)
    return fitness_lookup.get(bitstring, float('inf'))

def sigmoid(x):
    """Element-wise sigmoid function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

# --- BPSO Implementation ---

class Particle:
    """Represents a particle in the BPSO swarm."""
    def __init__(self, n_features, fitness_lookup):
        self.position = np.random.randint(0, 2, n_features)
        while array_to_bitstring(self.position) not in fitness_lookup:
             self.position = np.random.randint(0, 2, n_features)

        self.velocity = np.random.uniform(-V_MAX / 2, V_MAX / 2, n_features)
        #self.velocity = np.zeros(n_features)

        # Personal best initialised to start position
        self.pbest_position = self.position.copy()
        self.pbest_fitness = get_fitness(self.position, fitness_lookup)

def run_bpso(fitness_lookup, n_features):
    """Runs Binary Particle Swarm Optimization."""
    swarm = [Particle(n_features, fitness_lookup) for _ in range(SWARM_SIZE)]

    # Initialise global best
    gbest_fitness = float('inf')
    gbest_position = None
    for p in swarm:
        if p.pbest_fitness < gbest_fitness:
            gbest_fitness = p.pbest_fitness
            gbest_position = p.pbest_position.copy()

    if gbest_position is None: # If all started with inf fitness
         print("Warning: Could not initialise gbest. Check fitness_dict.")
         # Choose a random valid position as start-gbest
         valid_start_key = random.choice(list(fitness_lookup.keys()))
         gbest_position = np.array(list(valid_start_key), dtype=int)
         gbest_fitness = fitness_lookup[valid_start_key]


    fitness_history = []
    print(f"Starting BPSO: Swarm={SWARM_SIZE}, Iter={N_ITERATIONS}, N={n_features}")
    start_time = time.time()

    for iteration in range(N_ITERATIONS):
        # Linear decrease of inertia weight (common)
        w = W_MAX - (W_MAX - W_MIN) * iteration / N_ITERATIONS

        # Update each particle
        for particle in swarm:
            # Calculate new velocity
            r1 = np.random.rand(n_features)
            r2 = np.random.rand(n_features)
            cognitive_v = C1 * r1 * (particle.pbest_position - particle.position)
            social_v = C2 * r2 * (gbest_position - particle.position)
            particle.velocity = w * particle.velocity + cognitive_v + social_v

            # Limit velocity
            particle.velocity = np.clip(particle.velocity, -V_MAX, V_MAX)

            # Calculate probability for bit = 1
            probabilities = sigmoid(particle.velocity)

            # Update position based on probability
            random_values = np.random.rand(n_features)
            new_position = (random_values < probabilities).astype(int)

            # --- Important: Check if the new position is valid (exists in lookup) ---
            new_pos_bitstring = array_to_bitstring(new_position)
            if new_pos_bitstring in fitness_lookup:
                particle.position = new_position
                current_fitness = fitness_lookup[new_pos_bitstring]

                # Update particle's personal best (pbest)
                if current_fitness < particle.pbest_fitness:
                    particle.pbest_fitness = current_fitness
                    particle.pbest_position = particle.position.copy()

            # If new_pos_bitstring is not in fitness_lookup:
            # The particle will remain in its old position and pbest will not be updated.

        # Update global best (gbest) after all particles are updated
        for particle in swarm:
            if particle.pbest_fitness < gbest_fitness:
                gbest_fitness = particle.pbest_fitness
                gbest_position = particle.pbest_position.copy()

        fitness_history.append(gbest_fitness)

        # Print progress every 10 iterations
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration+1}/{N_ITERATIONS} - Best global fitness: {gbest_fitness:.6f}")

    end_time = time.time()
    print(f"BPSO completed in {end_time - start_time:.2f} seconds.")

    # Return best global solution (as bitstring), its fitness, and history
    gbest_bitstring = array_to_bitstring(gbest_position) if gbest_position is not None else "None"
    return gbest_bitstring, gbest_fitness, fitness_history

# --- Run BPSO and Display Results ---

if n_features > 0:
    gbest_bitstring_bpso, gbest_fitness_bpso, history_bpso = run_bpso(fitness_dict, n_features)

    print("\n--- Results from Binary PSO ---")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Best found solution (bitstring): {gbest_bitstring_bpso}")
    print(f"Best found fitness (h): {gbest_fitness_bpso:.6f}")

    # Compare with the actual global optimum
    global_optimum_fitness = df['fitness_h'].min()
    global_optimum_bitmask = df.loc[df['fitness_h'].idxmin(), 'subset_bitmask']
    print(f"\nActual global optimum (from table):")
    print(f"  Bitstring: {global_optimum_bitmask}")
    print(f"  Fitness (h): {global_optimum_fitness:.6f}")

    if np.isclose(gbest_fitness_bpso, global_optimum_fitness):
        print("\nBPSO found the global optimum (or a solution with equal fitness)!")
    else:
        print("\nBPSO found a local optimum (or stopped before convergence).")

    # Plot fitness history
    """ import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(range(N_ITERATIONS), history_bpso)
    plt.title(f"BPSO Fitness over Iterations ({DATASET_NAME})")
    plt.xlabel("Iteration")
    plt.ylabel("Best Global Fitness (h)")
    plt.grid(True)
    plt.ylim(bottom=max(0, global_optimum_fitness - 0.05), top=history_bpso[0] * 1.1) # Juster y-aksen
    plt.show() """

else:
    print("Could not run BPSO because N (number of features) was not determined correctly.")
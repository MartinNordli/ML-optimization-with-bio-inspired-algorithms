import numpy as np
import pandas as pd # Trengs egentlig ikke her lenger
import random
import math
import time
from collections import defaultdict
import matplotlib.pyplot as plt # For plotting i testblokk
import os
# --- PARAMETERE (kan overstyres fra run_all.py) ---
POPULATION_SIZE = 100
N_GENERATIONS = 150
CROSSOVER_PROB = 0.9
# MUTATION_PROB settes basert på n_features

# --- 3. Hjelpefunksjoner (Basis) ---
def array_to_bitstring(individual_array):
    """Konverterer numpy array [0, 1, 0] til bitstring '010'."""
    if not isinstance(individual_array, np.ndarray): return "Invalid"
    return "".join(individual_array.astype(str))

def get_objectives(individual_array, multi_objective_lookup):
    """Henter tuple med (error, num_features). Returnerer (inf, inf) hvis ikke funnet."""
    bitstring = array_to_bitstring(individual_array)
    return multi_objective_lookup.get(bitstring, (float('inf'), float('inf')))

# --- 4. NSGA-II Kjernefunksjoner ---

def dominates(obj1, obj2):
    """Sjekker om løsning 1 dominerer løsning 2 (begge mål minimeres)."""
    if obj1 == (float('inf'), float('inf')) or obj2 == (float('inf'), float('inf')):
         return False # Ugyldige løsninger dominerer ikke
    not_worse = all(o1 <= o2 for o1, o2 in zip(obj1, obj2))
    strictly_better = any(o1 < o2 for o1, o2 in zip(obj1, obj2))
    return not_worse and strictly_better

def fast_non_dominated_sort(objectives_list):
    """Utfører ikke-dominert sortering. Returnerer liste av fronter (lister av indekser)."""
    pop_size = len(objectives_list)
    S = [[] for _ in range(pop_size)] # Solutions dominated by p
    n = [0] * pop_size               # Domination count for p
    rank = [-1] * pop_size           # Rank (front number) for p
    fronts = [[]]                    # List of fronts F_1, F_2, ...

    for p_idx in range(pop_size):
        obj_p = objectives_list[p_idx]
        if obj_p[0] == float('inf') or obj_p[1] == float('inf'):
            n[p_idx] = -99 # Marker som helt ugyldig
            continue

        for q_idx in range(pop_size):
            if p_idx == q_idx: continue
            obj_q = objectives_list[q_idx]
            if obj_q[0] == float('inf') or obj_q[1] == float('inf'): continue

            if dominates(obj_p, obj_q):
                S[p_idx].append(q_idx)
            elif dominates(obj_q, obj_p):
                n[p_idx] += 1

        if n[p_idx] == 0: # p tilhører første front
            rank[p_idx] = 0
            fronts[0].append(p_idx)

    i = 0
    while fronts[i]:
        next_front = []
        for p_idx in fronts[i]:
            for q_idx in S[p_idx]:
                if n[q_idx] == -99: continue # Skip invalid solutions
                n[q_idx] -= 1
                if n[q_idx] == 0:
                    rank[q_idx] = i + 1
                    next_front.append(q_idx)
        i += 1
        if next_front:
            fronts.append(next_front)
        else:
            break
    return fronts, rank # Returner også ranks for enklere bruk

def crowding_distance_assignment(objectives_list, front_indices):
    """Beregner crowding distance for individer på en front."""
    if not front_indices: return {}

    # Filtrer objectives_list for å kun inneholde gyldige mål for front_indices
    valid_objectives_in_front = {idx: objectives_list[idx] for idx in front_indices if objectives_list[idx][0] != float('inf')}
    valid_front_indices = list(valid_objectives_in_front.keys())

    if not valid_front_indices: return {} # Ingen gyldige individer på fronten

    num_objectives = len(valid_objectives_in_front[valid_front_indices[0]])
    num_in_front = len(valid_front_indices)
    distances = {idx: 0.0 for idx in valid_front_indices}

    for m in range(num_objectives):
        # Sorter den gyldige fronten etter dette målet
        sorted_valid_front = sorted(valid_objectives_in_front.items(), key=lambda item: item[1][m])

        obj_min = sorted_valid_front[0][1][m]
        obj_max = sorted_valid_front[-1][1][m]
        obj_range = obj_max - obj_min

        # Sett uendelig avstand for ytterpunktene i den *gyldige* fronten
        distances[sorted_valid_front[0][0]] = float('inf')
        if num_in_front > 1: # Unngå indexfeil hvis bare ett element
             distances[sorted_valid_front[-1][0]] = float('inf')

        if obj_range == 0 or num_in_front <= 2: continue # Ingen avstand å beregne

        # Beregn avstand for punktene i midten
        for i in range(1, num_in_front - 1):
            idx_current = sorted_valid_front[i][0]
            # Hent mål fra den opprinnelige, komplette listen
            obj_next = objectives_list[sorted_valid_front[i+1][0]][m]
            obj_prev = objectives_list[sorted_valid_front[i-1][0]][m]
            distances[idx_current] += (obj_next - obj_prev) / obj_range

    return distances

# --- 5. Genetiske Operatorer (tilpasset MOEA) ---
def initialize_population_mo(pop_size, chromosome_length, multi_objective_lookup):
    """Lager startpopulasjon, sikrer gyldige individer."""
    population = []
    attempts = 0
    max_attempts = pop_size * 10
    while len(population) < pop_size and attempts < max_attempts:
        p = np.random.randint(0, 2, chromosome_length)
        if array_to_bitstring(p) in multi_objective_lookup:
             population.append(p)
        attempts += 1
    if len(population) < pop_size:
         print(f"ADVARSEL (NSGA Init): Klarte bare å initialisere {len(population)}/{pop_size} gyldige individer.")
    return population

# uniform_crossover (samme som for GA)
def uniform_crossover(parent1, parent2, pc):
    """Utfører uniform krysning med sannsynlighet pc."""
    if parent1 is None or parent2 is None: return parent1, parent2
    if random.random() < pc:
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        for i in range(len(parent1)):
            if random.random() < 0.5:
                offspring1[i], offspring2[i] = offspring2[i], offspring1[i]
        return offspring1, offspring2
    else:
        return parent1.copy(), parent2.copy()

# --- KORRIGERT MUTASJONSFUNKSJON ---
def bit_flip_mutation_mo(individual, pm, multi_objective_lookup):
    """Bit-flip mutasjon, sjekker gyldighet mot multi-objektiv lookup."""
    if individual is None: return None

    mutated_individual = individual.copy()
    for i in range(len(mutated_individual)):
        if random.random() < pm:
            mutated_individual[i] = 1 - mutated_individual[i]

    # Sjekk gyldighet mot den *mottatte* multi_objective_lookup
    if array_to_bitstring(mutated_individual) not in multi_objective_lookup:
         return individual # Returner originalen hvis mutasjon er ugyldig
    return mutated_individual

def crowded_comparison_operator(idx1, idx2, ranks, distances):
    """Sammenligner to individer basert på rank og crowding distance."""
    # ranks er nå en liste/array, ikke dict
    rank1 = ranks[idx1] if ranks[idx1] != -1 else float('inf')
    rank2 = ranks[idx2] if ranks[idx2] != -1 else float('inf')

    if rank1 < rank2:
        return idx1
    elif rank2 < rank1:
        return idx2
    else: # Samme rank
        dist1 = distances.get(idx1, -1) # Bruk negativ hvis mangler
        dist2 = distances.get(idx2, -1)
        if dist1 > dist2: # Større avstand er bedre
            return idx1
        elif dist2 > dist1:
            return idx2
        else:
            return random.choice([idx1, idx2])

# --- 6. Hoved NSGA-II Kjøring ---
def run_nsga2(multi_objective_lookup, n_features,
              pop_size=POPULATION_SIZE,
              n_generations=N_GENERATIONS,
              crossover_prob=CROSSOVER_PROB,
              mutation_prob_factor=1.5):
    """Kjører NSGA-II algoritmen."""
    mutation_prob = mutation_prob_factor / n_features

    population = initialize_population_mo(pop_size, n_features, multi_objective_lookup)
    if not population:
         print("FEIL (NSGA-II): Kunne ikke initialisere populasjonen.")
         return {'pareto_solutions': [], 'pareto_objectives': []}

    pop_objectives = [get_objectives(ind, multi_objective_lookup) for ind in population]

    print(f"  Starter NSGA-II: Pop={pop_size}, Gen={n_generations}, N={n_features}")
    # start_time = time.time() # Tidtaking i run_all.py

    for generation in range(n_generations):
        # Lag avkom Q(t)
        pop_fronts, pop_ranks = fast_non_dominated_sort(pop_objectives) # Få ranks
        pop_distances = {}
        for front in pop_fronts: # Beregn avstand for alle fronter
            pop_distances.update(crowding_distance_assignment(pop_objectives, front))

        offspring_population = []
        while len(offspring_population) < pop_size:
             # Crowded tournament selection
             p1_idx = random.randrange(len(population))
             p2_idx = random.randrange(len(population))
             parent1_idx = crowded_comparison_operator(p1_idx, p2_idx, pop_ranks, pop_distances)

             p3_idx = random.randrange(len(population))
             p4_idx = random.randrange(len(population))
             parent2_idx = crowded_comparison_operator(p3_idx, p4_idx, pop_ranks, pop_distances)

             parent1 = population[parent1_idx]
             parent2 = population[parent2_idx]

             offspring1, offspring2 = uniform_crossover(parent1, parent2, crossover_prob)
             # --- KORRIGERT KALL til mutasjon ---
             mutated_offspring1 = bit_flip_mutation_mo(offspring1, mutation_prob, multi_objective_lookup)
             mutated_offspring2 = bit_flip_mutation_mo(offspring2, mutation_prob, multi_objective_lookup)

             if mutated_offspring1 is not None:
                 offspring_population.append(mutated_offspring1)
             if len(offspring_population) < pop_size and mutated_offspring2 is not None:
                 offspring_population.append(mutated_offspring2)

        # Evaluer avkom
        offspring_objectives = [get_objectives(ind, multi_objective_lookup) for ind in offspring_population]

        # Kombiner og velg neste generasjon P(t+1)
        combined_population = population + offspring_population
        combined_objectives = pop_objectives + offspring_objectives

        combined_fronts, combined_ranks = fast_non_dominated_sort(combined_objectives)

        new_population = []
        new_pop_objectives = []
        remaining_capacity = pop_size
        front_num = 0

        while remaining_capacity > 0 and front_num < len(combined_fronts):
            current_front_indices = combined_fronts[front_num]
            if not current_front_indices:
                front_num += 1; continue

            num_on_front = len(current_front_indices)
            if num_on_front <= remaining_capacity:
                # Hele fronten får plass
                for idx in current_front_indices:
                    new_population.append(combined_population[idx])
                    new_pop_objectives.append(combined_objectives[idx])
                remaining_capacity -= num_on_front
                front_num += 1
            else:
                # Siste front må sorteres etter crowding distance
                distances_on_front = crowding_distance_assignment(combined_objectives, current_front_indices)
                sorted_last_front = sorted(current_front_indices, key=lambda idx: distances_on_front.get(idx, -1), reverse=True) # Negativ avstand hvis mangler
                for i in range(remaining_capacity):
                    idx = sorted_last_front[i]
                    new_population.append(combined_population[idx])
                    new_pop_objectives.append(combined_objectives[idx])
                remaining_capacity = 0 # Full

        population = new_population
        pop_objectives = new_pop_objectives

        # Skriv ut fremdrift
        # if (generation + 1) % 10 == 0:
        #     num_in_front0 = len(fast_non_dominated_sort(pop_objectives)[0][0]) if pop_objectives else 0
        #     print(f"    NSGA-II Gen {generation+1}/{n_generations} - Antall i Front 0: {num_in_front0}")


    # end_time = time.time()
    # print(f"  NSGA-II fullført på {end_time - start_time:.2f} sekunder.") # Tidtaking i run_all.py

    # Returner første front fra siste populasjon som dictionary
    final_fronts, _ = fast_non_dominated_sort(pop_objectives)
    pareto_front_indices = final_fronts[0] if final_fronts else []
    pareto_front_solutions = [population[i] for i in pareto_front_indices]
    pareto_front_objectives = [pop_objectives[i] for i in pareto_front_indices]

    result_dict = {
        'pareto_solutions': pareto_front_solutions,
        'pareto_objectives': pareto_front_objectives
    }
    return result_dict


# --- Standalone Kjøring ---
if __name__ == "__main__":
    # 1. Definer datasett for direkte kjøring
    STANDALONE_DATASET = "Wine" # <-- ENDRE HER for å teste
    STANDALONE_MODEL = "RandomForestClassifier"
    STANDALONE_TABLES_FOLDER = "tables"
    print(f"--- Kjører {__file__} direkte på datasett: {STANDALONE_DATASET} ---")

    # 2. Last inn data
    objectives_lookup_standalone = {}
    n_features_standalone = 0
    try:
        lookup_file = os.path.join(STANDALONE_TABLES_FOLDER, f"lookup_table_{STANDALONE_DATASET}_{STANDALONE_MODEL}.csv")
        if not os.path.exists(lookup_file): raise FileNotFoundError(f"Fant ikke {lookup_file}")

        df_standalone = pd.read_csv(lookup_file)
        required_cols = ['subset_bitmask', 'num_features', 'error_hE']
        if not all(col in df_standalone.columns for col in required_cols): raise ValueError("Mangler kolonner")

        df_standalone['subset_bitmask'] = df_standalone['subset_bitmask'].astype(str)
        n_features_standalone = int(df_standalone['num_features'].max())
        df_standalone['subset_bitmask'] = df_standalone['subset_bitmask'].str.zfill(n_features_standalone)
        df_standalone.dropna(subset=['error_hE', 'num_features'], inplace=True)
        if df_standalone.empty: raise ValueError("Tom DataFrame")

        objectives_lookup_standalone = {row['subset_bitmask']: (row['error_hE'], int(row['num_features']))
                                        for _, row in df_standalone.iterrows()}
        print(f"Data lastet: N={n_features_standalone}, Antall løsninger: {len(objectives_lookup_standalone)}")

    except Exception as e:
        print(f"FEIL under datalasting for standalone kjøring: {e}")
        exit()

    # 3. Kjør NSGA-II
    if n_features_standalone > 0:
        # Bruk færre generasjoner for raskere test
        result = run_nsga2(objectives_lookup_standalone, n_features_standalone, n_generations=50, pop_size=50)

        # 4. Skriv ut resultater
        print("\n--- Standalone NSGA-II Resultat ---")
        print(f"Antall løsninger funnet på fronten: {len(result.get('pareto_solutions', []))}")
        print("\nLøsninger (Error, Num Features):")
        pareto_objectives_standalone = result.get('pareto_objectives', [])
        pareto_solutions_standalone = result.get('pareto_solutions', [])
        pareto_sorted = sorted(zip(pareto_solutions_standalone, pareto_objectives_standalone), key=lambda x: x[1][1]) # Sorter etter features
        for sol_array, obj in pareto_sorted:
            print(f"  Bitstring: {array_to_bitstring(sol_array)} -> Error={obj[0]:.4f}, Features={int(obj[1])}")

        # 5. Plot fronten
        if pareto_objectives_standalone:
            errors = [obj[0] for obj in pareto_objectives_standalone]
            num_features_list = [obj[1] for obj in pareto_objectives_standalone]
            plt.figure(figsize=(10, 7))
            plt.scatter(num_features_list, errors, c='red', marker='x', label='Funnet Pareto Front (NSGA-II)')

            # Plot også den sanne fronten for sammenligning
            all_objs = list(objectives_lookup_standalone.values())
            true_fronts, _ = fast_non_dominated_sort(all_objs)
            if true_fronts:
                true_front_indices_all = true_fronts[0]
                true_pareto_objectives = [all_objs[i] for i in true_front_indices_all]
                true_errors = [obj[0] for obj in true_pareto_objectives]
                true_num_features = [obj[1] for obj in true_pareto_objectives]
                true_pareto_sorted = sorted(zip(true_num_features, true_errors))
                true_num_features_sorted = [nf for nf, err in true_pareto_sorted]
                true_errors_sorted = [err for nf, err in true_pareto_sorted]
                plt.plot(true_num_features_sorted, true_errors_sorted, 'bo-', markersize=4, alpha=0.6, label='Faktisk Pareto Front (fra Lookup)')

            plt.title(f'Standalone NSGA-II Pareto Front ({STANDALONE_DATASET})')
            plt.xlabel('Antall Funksjoner')
            plt.ylabel('Feilrate (Error h_E)')
            plt.grid(True)
            unique_nf_ticks = sorted(list(set([int(nf) for nf in num_features_list])))
            if unique_nf_ticks :
                step = max(1, int(np.ceil((max(unique_nf_ticks) - min(unique_nf_ticks)) / 10))) if max(unique_nf_ticks) > min(unique_nf_ticks) else 1
                plt.xticks(np.arange(min(unique_nf_ticks), max(unique_nf_ticks) + 1, step=step))
            plt.legend()
            plt.show()
    else:
        print("Kunne ikke kjøre NSGA-II, n_features ikke satt.")
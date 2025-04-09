import numpy as np
import pandas as pd # Trengs egentlig ikke her lenger, men skader ikke
import random
import time
import os
# --- PARAMETERE (kan overstyres fra run_all.py om nødvendig) ---
# Disse kan settes her som standardverdier, eller sendes inn som argumenter
# til run_simple_ga hvis du vil ha mer fleksibilitet.
POPULATION_SIZE = 50
N_GENERATIONS = 100
CROSSOVER_PROB = 0.8
# MUTATION_PROB settes ofte basert på n_features (1.0 / n_features)
TOURNAMENT_SIZE = 3
ELITISM_COUNT = 1

# --- 3. GA Funksjoner ---

def initialize_population(pop_size, chromosome_length, fitness_lookup):
    """Lager en tilfeldig startpopulasjon av binære numpy-arrays."""
    population = []
    attempts = 0
    max_attempts = pop_size * 10
    while len(population) < pop_size and attempts < max_attempts:
        p = np.random.randint(0, 2, chromosome_length)
        # Sjekk gyldighet mot den *mottatte* fitness_lookup
        if array_to_bitstring(p) in fitness_lookup:
             population.append(p)
        attempts += 1
    if len(population) < pop_size:
         print(f"ADVARSEL (GA Init): Klarte bare å initialisere {len(population)}/{pop_size} gyldige individer.")
    return population

def array_to_bitstring(individual_array):
    """Konverterer numpy array [0, 1, 0] til bitstring '010'."""
    # Sikrer at input ER en numpy array før konvertering
    if not isinstance(individual_array, np.ndarray):
        # Kan skje hvis f.eks. None blir sendt inn
        return "Invalid"
    return "".join(individual_array.astype(str))

def get_fitness(individual_array, fitness_lookup):
    """Henter fitness fra lookup-dictionary. Returnerer uendelig hvis ikke funnet."""
    bitstring = array_to_bitstring(individual_array)
    return fitness_lookup.get(bitstring, float('inf')) # Håndterer ukjente bitstrings

def tournament_selection(population, fitnesses, k):
    """Velger en vinner fra en tilfeldig turnering av størrelse k."""
    population_size = len(population)
    if population_size == 0: return None # Håndter tom populasjon
    if population_size < k: k = population_size # Juster k hvis populasjonen er liten
    # Velg k tilfeldige (unike) indekser
    tournament_indices = random.sample(range(population_size), k)
    # Finn indeksen til den beste (laveste fitness) blant disse
    try:
        best_index_in_tournament = min(tournament_indices, key=lambda i: fitnesses[i])
        return population[best_index_in_tournament] # Returner vinner-individet
    except (IndexError, TypeError):
         # Fallback hvis fitnesses er tom eller har feil format
         print(f"Advarsel: Problem i tournament selection (pop_size={population_size}, k={k})")
         return population[random.choice(tournament_indices)]


def uniform_crossover(parent1, parent2, pc):
    """Utfører uniform krysning med sannsynlighet pc."""
    # Sikre at foreldre er gyldige numpy arrays
    if parent1 is None or parent2 is None:
        return parent1, parent2 # Returner uendret hvis en forelder mangler

    if random.random() < pc:
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        for i in range(len(parent1)):
            if random.random() < 0.5: # Bytt bit med 50% sannsynlighet
                offspring1[i], offspring2[i] = offspring2[i], offspring1[i] # Swap
        return offspring1, offspring2
    else:
        # Ingen krysning, returner kopier av foreldrene
        return parent1.copy(), parent2.copy()

# --- KORRIGERT MUTASJONSFUNKSJON ---
def bit_flip_mutation(individual, pm, fitness_lookup):
    """Utfører bit-flip mutasjon med sannsynlighet pm per bit."""
    if individual is None: return None # Håndter None input

    mutated_individual = individual.copy()
    for i in range(len(mutated_individual)):
        if random.random() < pm:
            mutated_individual[i] = 1 - mutated_individual[i] # Flip bit

    # Sikre at det muterte individet finnes i lookup'en ved å bruke mottatt dict
    if array_to_bitstring(mutated_individual) not in fitness_lookup:
         return individual # Returner originalen hvis mutasjon ga ugyldig løsning
    return mutated_individual

# --- 4. Hoved GA Kjøring ---

def run_simple_ga(fitness_lookup, n_features,
                  pop_size=POPULATION_SIZE,
                  n_generations=N_GENERATIONS,
                  crossover_prob=CROSSOVER_PROB,
                  mutation_prob_factor=1.0, # Faktor for pm = factor / n_features
                  tournament_size=TOURNAMENT_SIZE,
                  elitism_count=ELITISM_COUNT):
    """Kjører den enkle genetiske algoritmen."""
    # Beregn mutasjonssannsynlighet
    mutation_prob = mutation_prob_factor / n_features

    population = initialize_population(pop_size, n_features, fitness_lookup)
    if not population:
         print("FEIL (GA): Kunne ikke initialisere populasjonen.")
         return "None", float('inf'), [] # Returner feilindikator

    best_overall_individual = None
    best_overall_fitness = float('inf')
    fitness_history = []

    print(f"  Starter GA: Pop={pop_size}, Gen={n_generations}, N={n_features}")
    # start_time = time.time() # Tidtaking kan gjøres i run_all.py

    for generation in range(n_generations):
        # Evaluer hele populasjonen
        fitnesses = [get_fitness(ind, fitness_lookup) for ind in population]

        # Finn beste i denne generasjonen og oppdater global beste
        # Håndter tilfelle der alle fitness er 'inf'
        valid_fitnesses = [f for f in fitnesses if f != float('inf')]
        if not valid_fitnesses:
             print(f"    Advarsel (GA Gen {generation}): Ingen gyldig fitness funnet i populasjonen.")
             # Kan velge å stoppe eller fortsette med tilfeldige handlinger
             current_best_fitness = float('inf')
        else:
             current_best_idx = np.argmin(fitnesses) # argmin håndterer inf greit
             current_best_fitness = fitnesses[current_best_idx]
             if current_best_fitness < best_overall_fitness:
                 best_overall_fitness = current_best_fitness
                 best_overall_individual = population[current_best_idx].copy()

        fitness_history.append(best_overall_fitness)

        # Lag neste generasjon
        new_population = []

        # Elitisme: Ta vare på de beste
        if elitism_count > 0 and valid_fitnesses: # Sjekk at det finnes gyldige fitnesses
            # Sorter indekser basert på fitness (lavest først)
            sorted_indices = np.argsort(fitnesses)
            for i in range(min(elitism_count, len(population))):
                idx = sorted_indices[i]
                # Unngå å legge til None hvis populasjonen av en eller annen grunn inneholder det
                if population[idx] is not None:
                    new_population.append(population[idx].copy())

        # Fyll resten av populasjonen ved reproduksjon
        while len(new_population) < pop_size:
            # Bruk try/except rundt seleksjon/krysning/mutasjon for robusthet
            try:
                 parent1 = tournament_selection(population, fitnesses, tournament_size)
                 parent2 = tournament_selection(population, fitnesses, tournament_size)

                 offspring1, offspring2 = uniform_crossover(parent1, parent2, crossover_prob)

                 # --- KORRIGERT KALL til mutasjon ---
                 mutated_offspring1 = bit_flip_mutation(offspring1, mutation_prob, fitness_lookup)
                 mutated_offspring2 = bit_flip_mutation(offspring2, mutation_prob, fitness_lookup)

                 if mutated_offspring1 is not None:
                      new_population.append(mutated_offspring1)
                 if len(new_population) < pop_size and mutated_offspring2 is not None:
                      new_population.append(mutated_offspring2)
            except Exception as e:
                 print(f"    FEIL i reproduksjonsløkke (GA Gen {generation}): {e}. Fortsetter...")
                 # Legg til en tilfeldig gyldig person for å unngå stopp
                 if len(new_population) < pop_size:
                     new_population.append(random.choice(population)) # Eller lag en ny random

        population = new_population # Bytt ut gammel populasjon

        # Skriv ut fremdrift av og til (valgfritt, kan fjernes)
        # if (generation + 1) % 10 == 0:
        #      print(f"    GA Gen {generation+1}/{n_generations} - Beste fitness: {best_overall_fitness:.6f}")

    # end_time = time.time()
    # print(f"  GA fullført på {end_time - start_time:.2f} sekunder.") # Tidtaking i run_all.py

    # Returner resultater som dictionary (enklere for run_all.py)
    best_solution_bitstring = array_to_bitstring(best_overall_individual) if best_overall_individual is not None else "None"
    result_dict = {
        'best_bitstring': best_solution_bitstring,
        'best_fitness': best_overall_fitness,
        'history': fitness_history
    }
    return result_dict

# --- Standalone Kjøring ---
if __name__ == "__main__":
    # 1. Definer datasett for direkte kjøring
    STANDALONE_DATASET = "Wine" # <-- ENDRE HER for å teste annet datasett
    STANDALONE_MODEL = "RandomForestClassifier"
    STANDALONE_TABLES_FOLDER = "tables" # Antar tabeller ligger her
    print(f"--- Kjører {__file__} direkte på datasett: {STANDALONE_DATASET} ---")

    # 2. Last inn data for standalone kjøring
    fitness_lookup_standalone = {}
    n_features_standalone = 0
    try:
        lookup_file = os.path.join(STANDALONE_TABLES_FOLDER, f"lookup_table_{STANDALONE_DATASET}_{STANDALONE_MODEL}.csv")
        if not os.path.exists(lookup_file): raise FileNotFoundError(f"Fant ikke {lookup_file}")

        df_standalone = pd.read_csv(lookup_file)
        required_cols = ['subset_bitmask', 'num_features', 'fitness_h']
        if not all(col in df_standalone.columns for col in required_cols): raise ValueError("Mangler kolonner")

        df_standalone['subset_bitmask'] = df_standalone['subset_bitmask'].astype(str)
        n_features_standalone = int(df_standalone['num_features'].max())
        df_standalone['subset_bitmask'] = df_standalone['subset_bitmask'].str.zfill(n_features_standalone)
        df_standalone.dropna(subset=['fitness_h'], inplace=True)
        if df_standalone.empty: raise ValueError("Tom DataFrame etter NaN-fjerning")

        fitness_lookup_standalone = pd.Series(df_standalone.fitness_h.values, index=df_standalone.subset_bitmask).to_dict()
        print(f"Data lastet: N={n_features_standalone}, Antall løsninger: {len(fitness_lookup_standalone)}")

    except Exception as e:
        print(f"FEIL under datalasting for standalone kjøring: {e}")
        exit()

    # 3. Kjør GA-funksjonen
    if n_features_standalone > 0:
        result = run_simple_ga(fitness_lookup_standalone, n_features_standalone)

        # 4. Skriv ut resultater
        print("\n--- Standalone GA Resultat ---")
        print(f"Beste funnet bitstring: {result.get('best_bitstring')}")
        print(f"Beste funnet fitness: {result.get('best_fitness', float('nan')):.6f}")

        # 5. Plot fitness historikk (valgfritt)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(result.get('history', []))
        plt.title(f"Standalone GA Fitness History ({STANDALONE_DATASET})")
        plt.xlabel("Generasjon")
        plt.ylabel("Beste Fitness (h)")
        plt.grid(True)
        plt.show()
    else:
        print("Kunne ikke kjøre GA, n_features ikke satt.")
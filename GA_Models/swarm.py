import numpy as np
import pandas as pd
import random
import math # For np.exp
import time

# --- 1. Last inn Lookup Table og Fitness Data ---
# (Bruk samme innlastingskode som i GA-skriptet for å få df, n_features, fitness_dict)

DATASET_NAME = "Heart_Disease_Statlog" # Eller Heart_Disease_Statlog, Seeds
MODEL_NAME = "RandomForestClassifier"
LOOKUP_TABLE_FILENAME = f"tables/lookup_table_{DATASET_NAME}_{MODEL_NAME}.csv"

print(f"Laster inn lookup-tabell for BPSO: {LOOKUP_TABLE_FILENAME}")
n_features = 0
try:
    df = pd.read_csv(LOOKUP_TABLE_FILENAME)
    if 'subset_bitmask' not in df.columns or 'num_features' not in df.columns or 'fitness_h' not in df.columns:
         raise ValueError("Nødvendige kolonner mangler i lookup-tabellen.")

    df['subset_bitmask'] = df['subset_bitmask'].astype(str)
    n_features = int(df['num_features'].max())
    df['subset_bitmask'] = df['subset_bitmask'].str.zfill(n_features)
    df.dropna(subset=['fitness_h'], inplace=True)
    if df.empty:
        raise ValueError("Ingen gyldige data etter fjerning av NaN.")

    fitness_dict = pd.Series(df.fitness_h.values, index=df.subset_bitmask).to_dict()
    print(f"Lookup-tabell lastet for {DATASET_NAME}. N={n_features}. Antall løsninger: {len(fitness_dict)}")

except Exception as e:
    print(f"FEIL under lasting av lookup-tabell: {e}")
    exit()

# --- 2. BPSO Parametere ---
SWARM_SIZE = 30        # Antall partikler i svermen
# N_FEATURES = n_features (definert over)
N_ITERATIONS = 100     # Antall iterasjoner (generasjoner)
W_MAX = 0.9          # Startverdi for treghetsvekt (inertia weight)
W_MIN = 0.4          # Sluttverdi for treghetsvekt
C1 = 2.0             # Kognitiv konstant (tiltrekning mot pbest)
C2 = 2.0             # Sosial konstant (tiltrekning mot gbest)
V_MAX = 4.0            # Maksimal absoluttverdi for hastighet

# --- 3. Hjelpefunksjoner ---

def array_to_bitstring(individual_array):
    """Konverterer numpy array [0, 1, 0] til bitstring '010'."""
    return "".join(individual_array.astype(str))

def get_fitness(individual_array, fitness_lookup):
    """Henter fitness fra lookup-dictionary. Returnerer uendelig hvis ikke funnet."""
    bitstring = array_to_bitstring(individual_array)
    return fitness_lookup.get(bitstring, float('inf'))

def sigmoid(x):
    """Element-vis sigmoid funksjon."""
    # Bruker np.clip for å unngå overflow i np.exp
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

# --- 4. BPSO Implementasjon ---

class Particle:
    """Representerer en partikkel i BPSO-svermen."""
    def __init__(self, n_features, fitness_lookup):
        # Tilfeldig startposisjon (sikrer at den er gyldig)
        self.position = np.random.randint(0, 2, n_features)
        while array_to_bitstring(self.position) not in fitness_lookup:
             self.position = np.random.randint(0, 2, n_features)

        # Tilfeldig starthastighet (eller start med nuller)
        self.velocity = np.random.uniform(-V_MAX / 2, V_MAX / 2, n_features)
        #self.velocity = np.zeros(n_features)

        # Personlig beste initialiseres til startposisjonen
        self.pbest_position = self.position.copy()
        self.pbest_fitness = get_fitness(self.position, fitness_lookup)

def run_bpso(fitness_lookup, n_features):
    """Kjører Binary Particle Swarm Optimization."""
    # Initialiser svermen
    swarm = [Particle(n_features, fitness_lookup) for _ in range(SWARM_SIZE)]

    # Initialiser global beste
    gbest_fitness = float('inf')
    gbest_position = None
    for p in swarm:
        if p.pbest_fitness < gbest_fitness:
            gbest_fitness = p.pbest_fitness
            gbest_position = p.pbest_position.copy()

    if gbest_position is None: # Hvis alle startet med inf fitness
         print("Advarsel: Kunne ikke initialisere gbest. Sjekk fitness_dict.")
         # Velg en tilfeldig gyldig posisjon som start-gbest
         valid_start_key = random.choice(list(fitness_lookup.keys()))
         gbest_position = np.array(list(valid_start_key), dtype=int)
         gbest_fitness = fitness_lookup[valid_start_key]


    fitness_history = [] # Valgfritt: spore beste fitness
    print(f"Starter BPSO: Swarm={SWARM_SIZE}, Iter={N_ITERATIONS}, N={n_features}")
    start_time = time.time()

    for iteration in range(N_ITERATIONS):
        # Lineær nedtrapping av treghetsvekt (vanlig)
        w = W_MAX - (W_MAX - W_MIN) * iteration / N_ITERATIONS

        # Oppdater hver partikkel
        for particle in swarm:
            # Beregn ny hastighet
            r1 = np.random.rand(n_features)
            r2 = np.random.rand(n_features)
            cognitive_v = C1 * r1 * (particle.pbest_position - particle.position)
            social_v = C2 * r2 * (gbest_position - particle.position)
            particle.velocity = w * particle.velocity + cognitive_v + social_v

            # Begrens hastigheten
            particle.velocity = np.clip(particle.velocity, -V_MAX, V_MAX)

            # Beregn sannsynlighet for bit = 1
            probabilities = sigmoid(particle.velocity)

            # Oppdater posisjon basert på sannsynlighet
            random_values = np.random.rand(n_features)
            new_position = (random_values < probabilities).astype(int)

            # --- Viktig: Sjekk om den nye posisjonen er gyldig (finnes i lookup) ---
            new_pos_bitstring = array_to_bitstring(new_position)
            if new_pos_bitstring in fitness_lookup:
                # Kun oppdater posisjon og evaluer hvis den er gyldig
                particle.position = new_position
                current_fitness = fitness_lookup[new_pos_bitstring]

                # Oppdater partikkelens personlige beste (pbest)
                if current_fitness < particle.pbest_fitness:
                    particle.pbest_fitness = current_fitness
                    particle.pbest_position = particle.position.copy()

            # Hvis new_pos_bitstring IKKE er i fitness_lookup:
            # Partikkelen blir stående i sin gamle posisjon og pbest endres ikke.

        # Oppdater global beste (gbest) etter at alle partikler er oppdatert
        for particle in swarm:
            if particle.pbest_fitness < gbest_fitness:
                gbest_fitness = particle.pbest_fitness
                gbest_position = particle.pbest_position.copy()

        fitness_history.append(gbest_fitness)

        # Skriv ut fremdrift av og til
        if (iteration + 1) % 10 == 0:
            print(f"Iterasjon {iteration+1}/{N_ITERATIONS} - Beste globale fitness: {gbest_fitness:.6f}")

    end_time = time.time()
    print(f"BPSO fullført på {end_time - start_time:.2f} sekunder.")

    # Returner beste globale løsning (som bitstring), dens fitness, og historikk
    gbest_bitstring = array_to_bitstring(gbest_position) if gbest_position is not None else "None"
    return gbest_bitstring, gbest_fitness, fitness_history

# --- 5. Kjør BPSO og Vis Resultater ---

if n_features > 0:
    gbest_bitstring_bpso, gbest_fitness_bpso, history_bpso = run_bpso(fitness_dict, n_features)

    print("\n--- Resultat fra Binary PSO ---")
    print(f"Datasett: {DATASET_NAME}")
    print(f"Beste funnet løsning (bitstring): {gbest_bitstring_bpso}")
    print(f"Beste funnet fitness (h): {gbest_fitness_bpso:.6f}")

    # Sammenlign med det faktiske globale optimumet
    global_optimum_fitness = df['fitness_h'].min()
    global_optimum_bitmask = df.loc[df['fitness_h'].idxmin(), 'subset_bitmask']
    print(f"\nFaktisk globalt optimum (fra tabell):")
    print(f"  Bitstring: {global_optimum_bitmask}")
    print(f"  Fitness (h): {global_optimum_fitness:.6f}")

    if np.isclose(gbest_fitness_bpso, global_optimum_fitness): # Bruk isclose for flyttall
        print("\nBPSO fant det globale optimumet (eller en løsning med lik fitness)!")
    else:
        print("\nBPSO fant et lokalt optimum (eller stoppet før konvergens).")

    # Valgfritt: Plotte fitness-historikken
    """ import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(range(N_ITERATIONS), history_bpso)
    plt.title(f"BPSO Fitness over Iterasjoner ({DATASET_NAME})")
    plt.xlabel("Iterasjon")
    plt.ylabel("Beste Globale Fitness (h)")
    plt.grid(True)
    plt.ylim(bottom=max(0, global_optimum_fitness - 0.05), top=history_bpso[0] * 1.1) # Juster y-aksen
    plt.show() """

else:
    print("Kunne ikke kjøre BPSO fordi N (antall features) ikke ble bestemt korrekt.")
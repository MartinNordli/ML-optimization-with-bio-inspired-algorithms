import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Konfigurasjon ---
DATASET_NAME = "Heart_Disease_Statlog" # Endre for Heart_Disease_Statlog eller Seeds
MODEL_NAME = "RandomForestClassifier"
LOOKUP_TABLE_FILENAME = f"lookup_table_{DATASET_NAME}_{MODEL_NAME}.csv"
OUTPUT_PLOT_FILENAME = f"fitness_landscape_{DATASET_NAME}_{MODEL_NAME}.png"

# --- 1. Last inn Lookup-tabellen ---
print(f"Laster inn lookup-tabell: {LOOKUP_TABLE_FILENAME}")
n_features = 0 # Initialiser n_features
try:
    # Les CSV
    df = pd.read_csv(LOOKUP_TABLE_FILENAME)

    # --- START KORREKSJON ---
    # Sjekk om nødvendige kolonner finnes
    if 'subset_bitmask' not in df.columns:
        print("FEIL: Kolonnen 'subset_bitmask' finnes ikke i CSV-filen.")
        exit()
    if 'num_features' not in df.columns:
         print("FEIL: Kolonnen 'num_features' finnes ikke i CSV-filen.")
         exit()

    # Konverter 'subset_bitmask' til streng
    df['subset_bitmask'] = df['subset_bitmask'].astype(str)

    # Bestem N (antall features) fra den maksimale verdien i 'num_features'
    # Dette er tryggere enn å stole på lengden av den første bitmasken
    n_features = int(df['num_features'].max())
    print(f"Bestemte totalt antall funksjoner (N) til: {n_features} (basert på maks 'num_features')")

    # Legg til ledende nuller til bitmasken for å sikre korrekt lengde (N)
    df['subset_bitmask'] = df['subset_bitmask'].str.zfill(n_features)
    print(f"Konverterte 'subset_bitmask' til streng og sikret padding til {n_features} siffer.")
    # --- SLUTT KORREKSJON ---

except FileNotFoundError:
    print(f"FEIL: Filen {LOOKUP_TABLE_FILENAME} ble ikke funnet.")
    exit()
except Exception as e:
    print(f"FEIL under lasting eller konvertering av CSV: {e}")
    exit()


print(f"Tabell lastet med {len(df)} rader.")

# Sjekk for NaN i fitness (kan oppstå ved feil under generering)
if df['fitness_h'].isnull().any():
    print("Advarsel: Fjerner rader med NaN i 'fitness_h'.")
    df.dropna(subset=['fitness_h'], inplace=True)
    print(f"Antall rader etter fjerning av NaN: {len(df)}")

if df.empty:
    print("FEIL: Ingen gyldige data igjen i tabellen etter fjerning av NaN.")
    exit()

# --- 2. Forbered for raskt oppslag ---
# Bruker en dictionary for raskere oppslag
print("Forbereder fitness-oppslag...")
# Bruker den korrigerte df til å lage dictionary
fitness_dict = pd.Series(df.fitness_h.values, index=df.subset_bitmask).to_dict()
# Nå brukes n_features bestemt under innlasting

# --- 3. Identifiser Lokale Optima (Hamming-distanse 1) ---
print(f"Identifiserer lokale optima (N={n_features})...")
local_optima_indices = []

for index, row in tqdm(df.iterrows(), total=len(df), desc="Sjekker optima"):
    current_bitmask = row['subset_bitmask']
    current_fitness = row['fitness_h']
    is_local_optimum = True # Anta at det er et optimum inntil motsatt er bevist

    # Generer N naboer ved å flippe én bit om gangen
    for i in range(n_features):
        # Lag nabomasken ved å flippe bit i
        neighbor_mask_list = list(current_bitmask)
        neighbor_mask_list[i] = '1' if current_bitmask[i] == '0' else '0'
        neighbor_mask = "".join(neighbor_mask_list)

        # Hent naboens fitness fra dictionary (raskere)
        neighbor_fitness = fitness_dict.get(neighbor_mask)

        # Hvis naboen finnes OG dens fitness er BEDRE (lavere) eller lik,
        # er dette IKKE et (strengt) lokalt optimum.
        if neighbor_fitness is not None and neighbor_fitness <= current_fitness:
            # For å inkludere platåer som lokale optima, bruk '<' istedenfor '<='
            is_local_optimum = False
            break # Ingen grunn til å sjekke flere naboer

    if is_local_optimum:
        local_optima_indices.append(index)

# Legg til kolonne i DataFrame
df['is_local_optimum'] = False
df.loc[local_optima_indices, 'is_local_optimum'] = True

num_local_optima = df['is_local_optimum'].sum()
print(f"Antall lokale optima funnet: {num_local_optima}")

# --- 4. Identifiser Globalt Optimum ---
# idxmin() finner indeksen til den første forekomsten av minimumsverdien
global_optimum_idx = df['fitness_h'].idxmin()
global_optimum_fitness = df.loc[global_optimum_idx, 'fitness_h']
global_optimum_features = df.loc[global_optimum_idx, 'num_features']
print(f"Globalt optimum funnet ved {global_optimum_features} funksjoner med fitness {global_optimum_fitness:.6f}")

# Marker globalt optimum spesifikt
df['is_global_optimum'] = False
df.loc[global_optimum_idx, 'is_global_optimum'] = True
# Sørg for at det globale optimumet ikke også telles som et "vanlig" lokalt optimum i plotte-logikken
# Hvis det globale også er lokalt (som det bør være), unngå dobbel plotting
# df.loc[global_optimum_idx, 'is_local_optimum'] = False # Kan settes False for å unngå overlapp i plot

# --- 5. Plott Resultatene ---
print("Lager plot...")
plt.style.use('seaborn-v0_8-whitegrid') # Fin stil
plt.figure(figsize=(12, 8))

# Plott alle punkter først (med lav alpha/gjennomsiktighet)
plt.scatter(df['num_features'], df['fitness_h'],
            alpha=0.1, s=15, label='Alle løsninger', c='lightgray')

# Plott lokale optima (ikke globalt)
local_opt_df = df[df['is_local_optimum'] & ~df['is_global_optimum']]
plt.scatter(local_opt_df['num_features'], local_opt_df['fitness_h'],
            alpha=0.8, s=40, label=f'Lokale Optima ({len(local_opt_df)})', c='orange', marker='^')

# Plott globalt optimum
global_opt_df = df[df['is_global_optimum']]
plt.scatter(global_opt_df['num_features'], global_opt_df['fitness_h'],
            s=100, label=f'Globalt Optimum ({len(global_opt_df)})', c='red', marker='*', edgecolors='black')

# Tittel og akse-etiketter
plt.title(f'Fitness Landskap ({DATASET_NAME} - {MODEL_NAME})', fontsize=16)
plt.xlabel('Antall Funksjoner', fontsize=12)
plt.ylabel('Fitness (h = Error + Penalty)', fontsize=12)
plt.xticks(range(0, n_features + 1)) # Sikrer heltall på x-aksen
plt.legend(fontsize=10)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Lagre plot til fil
plt.savefig(OUTPUT_PLOT_FILENAME, dpi=300)
print(f"Plot lagret som: {OUTPUT_PLOT_FILENAME}")

# Vis plot (valgfritt, fjern hvis du kjører mange på rad)
plt.show()
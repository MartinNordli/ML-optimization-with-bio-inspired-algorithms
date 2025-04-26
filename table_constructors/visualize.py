import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Configuration ---
DATASET_NAME = "Heart_Disease_Statlog" # Change this according to the dataset
MODEL_NAME = "RandomForestClassifier"
LOOKUP_TABLE_FILENAME = f"lookup_table_{DATASET_NAME}_{MODEL_NAME}.csv"
OUTPUT_PLOT_FILENAME = f"fitness_landscape_{DATASET_NAME}_{MODEL_NAME}.png"

# --- Load the lookup table ---
print(f"Loading lookup table: {LOOKUP_TABLE_FILENAME}")
n_features = 0 # Initialize n_features
try:
    df = pd.read_csv(LOOKUP_TABLE_FILENAME)

    # --- START CORRECTION ---
    # Check if necessary columns exist
    if 'subset_bitmask' not in df.columns:
        print("ERROR: The column 'subset_bitmask' does not exist in the CSV file.")
        exit()
    if 'num_features' not in df.columns:
         print("ERROR: The column 'num_features' does not exist in the CSV file.")
         exit()

    # Convert 'subset_bitmask' to string
    df['subset_bitmask'] = df['subset_bitmask'].astype(str)

    # Determine N (number of features) from the maximum value in 'num_features'
    # This is safer than relying on the length of the first bitmask
    n_features = int(df['num_features'].max())
    print(f"Determined total number of features (N) to be: {n_features} (based on max 'num_features')")

    # Add leading zeros to the bitmask to ensure correct length (N)
    df['subset_bitmask'] = df['subset_bitmask'].str.zfill(n_features)
    print(f"Converted 'subset_bitmask' to string and ensured padding to {n_features} digits.")

except FileNotFoundError:
    print(f"ERROR: The file {LOOKUP_TABLE_FILENAME} was not found.")
    exit()
except Exception as e:
    print(f"ERROR during loading or converting the CSV: {e}")
    exit()


print(f"Table loaded with {len(df)} rows.")

# Check for NaN in fitness (can occur during generation)
if df['fitness_h'].isnull().any():
    print("Warning: Removing rows with NaN in 'fitness_h'.")
    df.dropna(subset=['fitness_h'], inplace=True)
    print(f"Number of rows after removing NaN: {len(df)}")

if df.empty:
    print("ERROR: No valid data left in the table after removing NaN.")
    exit()

# --- Prepare for fast lookup ---
print("Preparing fitness lookup...")
# Dictionary allows for faster lookup
fitness_dict = pd.Series(df.fitness_h.values, index=df.subset_bitmask).to_dict()

# --- Identify Local Optima (Hamming distance 1) ---
print(f"Identifying local optima (N={n_features})...")
local_optima_indices = []

for index, row in tqdm(df.iterrows(), total=len(df), desc="Sjekker optima"):
    current_bitmask = row['subset_bitmask']
    current_fitness = row['fitness_h']
    is_local_optimum = True # Assume it is an optimum until proven otherwise

    # Generate N neighbors by flipping one bit at a time
    for i in range(n_features):
        # Create the neighbor mask by flipping the bit
        neighbor_mask_list = list(current_bitmask)
        neighbor_mask_list[i] = '1' if current_bitmask[i] == '0' else '0'
        neighbor_mask = "".join(neighbor_mask_list)
        neighbor_fitness = fitness_dict.get(neighbor_mask)

        # If the neighbor exists and its fitness is better (lower) or equal,
        # then this is NOT a (strict) local optimum.
        if neighbor_fitness is not None and neighbor_fitness <= current_fitness:
            # To include flat areas as local optima, we could use '<' instead of '<='
            is_local_optimum = False
            break

    if is_local_optimum:
        local_optima_indices.append(index)

# Add column to DataFrame
df['is_local_optimum'] = False
df.loc[local_optima_indices, 'is_local_optimum'] = True

num_local_optima = df['is_local_optimum'].sum()
print(f"Number of local optima found: {num_local_optima}")

# --- Identify Global Optimum ---
# idxmin() finds the index of the first occurrence of the minimum value
global_optimum_idx = df['fitness_h'].idxmin()
global_optimum_fitness = df.loc[global_optimum_idx, 'fitness_h']
global_optimum_features = df.loc[global_optimum_idx, 'num_features']
print(f"Globalt optimum funnet ved {global_optimum_features} funksjoner med fitness {global_optimum_fitness:.6f}")

# Mark the global optimum specifically
df['is_global_optimum'] = False
df.loc[global_optimum_idx, 'is_global_optimum'] = True

# --- Plot the results ---
print("Creating plot...")
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(12, 8))
plt.scatter(df['num_features'], df['fitness_h'],
            alpha=0.1, s=15, label='All solutions', c='lightgray')

# Plot local optima (not global)
local_opt_df = df[df['is_local_optimum'] & ~df['is_global_optimum']]
plt.scatter(local_opt_df['num_features'], local_opt_df['fitness_h'],
            alpha=0.8, s=40, label=f'Local Optima ({len(local_opt_df)})', c='orange', marker='^')

# Plot global optimum
global_opt_df = df[df['is_global_optimum']]
plt.scatter(global_opt_df['num_features'], global_opt_df['fitness_h'],
            s=100, label=f'Global Optimum ({len(global_opt_df)})', c='red', marker='*', edgecolors='black')

plt.title(f'Fitness Landscape ({DATASET_NAME} - {MODEL_NAME})', fontsize=16)
plt.xlabel('Number of Features', fontsize=12)
plt.ylabel('Fitness (h = Error + Penalty)', fontsize=12)
plt.xticks(range(0, n_features + 1))
plt.legend(fontsize=10)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Save plot to file
plt.savefig(OUTPUT_PLOT_FILENAME, dpi=300)
print(f"Plot saved as: {OUTPUT_PLOT_FILENAME}")
plt.show()
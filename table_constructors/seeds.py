import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
DATASET_NAME = "Seeds"
OPENML_ID = 1499 
MODEL_NAME = "RandomForestClassifier"
EPSILON = 0.01
N_ESTIMATORS = 100
TEST_SIZE = 0.3
RANDOM_STATE = 42
OUTPUT_FILENAME = f"tables/lookup_table_{DATASET_NAME}_{MODEL_NAME}.csv"
MAX_COMBINATIONS = None

# --- Load and prepare data ---
print(f"Loading data for: {DATASET_NAME} (OpenML ID: {OPENML_ID})...")
X = None
y = None
feature_names = None
try:
    seeds_data = fetch_openml(data_id=OPENML_ID, as_frame=False, parser='liac-arff')
    X = seeds_data.data
    y_raw = seeds_data.target
    try:
      feature_names = seeds_data.feature_names
    except AttributeError:
      feature_names = [f'feature_{i}' for i in range(X.shape[1])]


    # --- Preprocessing of y for Seeds ---
    unique_targets_raw = np.unique(y_raw)
    print(f"Unique values found in raw target (y_raw): {unique_targets_raw}")

    if np.all(np.isin(unique_targets_raw, ['1', '2', '3'])):
        print("Mapping target: '1' -> 0, '2' -> 1, '3' -> 2")
        mapper = {'1': 0, '2': 1, '3': 2}
        y = np.array([mapper[val] for val in y_raw])
    elif np.all(np.isin(unique_targets_raw, [1.0, 2.0, 3.0])): # If floats
        print("Mapping target: 1.0 -> 0, 2.0 -> 1, 3.0 -> 2")
        mapper = {1.0: 0, 2.0: 1, 3.0: 2}
        y = np.array([mapper[val] for val in y_raw])
    else:
        print(f"Warning: Unknown format for target values: {unique_targets_raw}. Trying direct conversion to int.")
        try:
            y = y_raw.astype(int)
            if np.min(y) > 0:
                 print(f"Adjusting y ({np.unique(y)}) to start at 0.")
                 y = y - np.min(y)
        except Exception as map_e:
             print(f"Error during mapping of y: {map_e}. Cannot continue.")
             exit()

    print(f"Unique values in final target (y): {np.unique(y)}")
    # Check that we have 3 classes (0, 1, 2)
    if len(np.unique(y)) != 3:
         print(f"Warning: Expected 3 classes (0, 1, 2), but found {len(np.unique(y))}. Check the mapping.")

    print("Data hentet successfully.")

    # Check for NaN
    if np.isnan(X).any():
        print("Warning: NaN (missing values) found in X. Requires handling.")

except Exception as e:
    print(f"Error: Could not fetch or process data from OpenML: {e}")
    print("Check internet connection, OpenML ID (1499), and the format of the data.")
    exit()

n_features = X.shape[1]
print(f"Dataset loaded: {DATASET_NAME} with {n_features} features.")
print(f"Feature names: {feature_names}")


# --- Split data once ---
print("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# --- Define the model ---
rf_model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    class_weight='balanced',
    n_jobs=-1
)

# --- Iterate through all feature combinations ---
results = []
total_combinations = 2**n_features - 1 # Will be 127 for N=7
limit_combinations = MAX_COMBINATIONS if MAX_COMBINATIONS is not None else total_combinations

print(f"\nStarting evaluation of feature subsets for {MODEL_NAME} on {DATASET_NAME}...")
print(f"Number of features (N): {n_features}")
print(f"Total number of possible non-empty combinations: {total_combinations:,}")
if MAX_COMBINATIONS is not None:
    print(f"LIMITED to run max {limit_combinations:,} combinations.")
else:
    print(f"Running all {limit_combinations:,} combinations (N=7). This is fast!")

start_time = time.time()

for i in tqdm(range(1, limit_combinations + 1), total=limit_combinations, desc=f"Evaluating {DATASET_NAME}"):
    bitmask = bin(i)[2:].zfill(n_features)
    selected_indices = [idx for idx, bit in enumerate(bitmask) if bit == '1']

    if not selected_indices: continue

    num_selected_features = len(selected_indices)
    X_train_subset = X_train[:, selected_indices]
    X_test_subset = X_test[:, selected_indices]

    try:
        # Train, predict, evaluate
        rf_model.fit(X_train_subset, y_train)
        y_pred_subset = rf_model.predict(X_test_subset)
        accuracy = accuracy_score(y_test, y_pred_subset)
        h_E = 1.0 - accuracy
        h_P = EPSILON * num_selected_features
        h = h_E + h_P

        # Save results
        results.append({
            'subset_indices': tuple(selected_indices), 'subset_bitmask': bitmask,
            'num_features': num_selected_features, 'accuracy': accuracy,
            'error_hE': h_E, 'penalty_hP': h_P, 'fitness_h': h
        })

    except Exception as e:
        print(f"\nError during evaluation of combination {i} (bitmask: {bitmask}): {e}")
        results.append({
             'subset_indices': tuple(selected_indices), 'subset_bitmask': bitmask,
             'num_features': num_selected_features, 'accuracy': np.nan,
             'error_hE': np.nan, 'penalty_hP': np.nan, 'fitness_h': np.nan
        })

# --- Save results ---
print("\nEvaluation complete. Converting results to DataFrame...")
lookup_table_df = pd.DataFrame(results)

print(f"Saving lookup table to file: {OUTPUT_FILENAME}")
lookup_table_df.to_csv(OUTPUT_FILENAME, index=False)

end_time = time.time()
total_time = end_time - start_time

print("\n--- Summary ---")
print(f"Dataset: {DATASET_NAME}")
print(f"Model: {MODEL_NAME}")
print(f"Number of evaluated combinations: {len(results)} of {total_combinations}")
print(f"Lookup table saved as: {OUTPUT_FILENAME}")
print(f"Total time taken: {total_time:.2f} seconds")
print("\nFirst 5 rows of lookup table (unsorted):")
print(lookup_table_df.head())

# Find and display the best combination
if not lookup_table_df['fitness_h'].isnull().all():
    best_row = lookup_table_df.loc[lookup_table_df['fitness_h'].idxmin()]
    print("\nBest combination found (lowest h):")
    print(f"  Fitness (h): {best_row['fitness_h']:.6f}")
    print(f"  Error (h_E): {best_row['error_hE']:.6f} (Accuracy: {best_row['accuracy']:.6f})")
    print(f"  Penalty (h_P): {best_row['penalty_hP']:.6f}")
    print(f"  Number of features: {best_row['num_features']}")
    print(f"  Bitmask: {best_row['subset_bitmask']}")

else:
    print("\nNo valid fitness values found.")
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import time

# --- Configuration ---
DATASET_NAME = "Wine"
MODEL_NAME = "RandomForestClassifier"
EPSILON = 0.01
N_ESTIMATORS = 100
TEST_SIZE = 0.3
RANDOM_STATE = 42
OUTPUT_FILENAME = f"tables/lookup_table_{DATASET_NAME}_{MODEL_NAME}.csv"
MAX_COMBINATIONS = None

# --- Load and prepare data ---
print(f"Loading data for: {DATASET_NAME}...")
dataset = load_wine()
X = dataset.data
y = dataset.target
try:
    feature_names = dataset.feature_names
except AttributeError:
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]

n_features = X.shape[1]
print(f"Datasett lastet: {DATASET_NAME} med {n_features} funksjoner.")
print(f"Funksjonsnavn: {feature_names}")

# --- Split data ---
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
total_combinations = 2**n_features - 1
limit_combinations = MAX_COMBINATIONS if MAX_COMBINATIONS is not None else total_combinations

print(f"\nStarting evaluation of feature subsets for {MODEL_NAME} on {DATASET_NAME}...")
print(f"Selected error metric (h_E): 1 - Accuracy")
print(f"Selected penalty (h_P): Epsilon * (Number of features), with Epsilon = {EPSILON}")
print(f"Total fitness (h): h_E + h_P (Minimized)")
print(f"Number of features (N): {n_features}")
print(f"Total number of possible non-empty combinations: {total_combinations:,}")
if MAX_COMBINATIONS is not None:
    print(f"LIMITED to run max {limit_combinations:,} combinations for testing.")
else:
    print(f"Running all {limit_combinations:,} combinations. For N=13, this can take a few minutes to hours.")

start_time = time.time()

for i in tqdm(range(1, limit_combinations + 1), total=limit_combinations, desc=f"Evaluating {DATASET_NAME}"):
    bitmask = bin(i)[2:].zfill(n_features)
    selected_indices = [idx for idx, bit in enumerate(bitmask) if bit == '1']

    if not selected_indices:
        continue

    num_selected_features = len(selected_indices)

    X_train_subset = X_train[:, selected_indices]
    X_test_subset = X_test[:, selected_indices]

    try:
        rf_model.fit(X_train_subset, y_train)
        y_pred_subset = rf_model.predict(X_test_subset)
        accuracy = accuracy_score(y_test, y_pred_subset)
        h_E = 1.0 - accuracy
        h_P = EPSILON * num_selected_features
        h = h_E + h_P

        results.append({
            'subset_indices': tuple(selected_indices),
            'subset_bitmask': bitmask,
            'num_features': num_selected_features,
            'accuracy': accuracy,
            'error_hE': h_E,
            'penalty_hP': h_P,
            'fitness_h': h
        })

    except Exception as e:
        print(f"\nERROR during evaluation of combination {i} (bitmask: {bitmask}): {e}")
        results.append({
            'subset_indices': tuple(selected_indices),
            'subset_bitmask': bitmask,
            'num_features': num_selected_features,
            'accuracy': np.nan, 'error_hE': np.nan, 'penalty_hP': np.nan, 'fitness_h': np.nan
        })


print("\nEvaluation complete (or stopped). Converting results to DataFrame...")
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
print(f"Total time taken: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print("\nFirst 5 rows of lookup table (unsorted):")
print(lookup_table_df.head())

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
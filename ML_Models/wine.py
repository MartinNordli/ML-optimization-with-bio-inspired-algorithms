import pandas as pd
import numpy as np
# Endret import for å laste Wine-datasettet
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm # For fremdriftslinje
import time # For tidsmåling

# --- Konfigurasjon ---
DATASET_NAME = "Wine"
MODEL_NAME = "RandomForestClassifier"
EPSILON = 0.01
N_ESTIMATORS = 100
TEST_SIZE = 0.3
RANDOM_STATE = 42
OUTPUT_FILENAME = f"tables/lookup_table_{DATASET_NAME}_{MODEL_NAME}.csv"
# Sett til None for å kjøre alle 8191 kombinasjoner for N=13
# Sett til f.eks. 100 for en rask test
MAX_COMBINATIONS = None

# --- 1. Last inn og forbered data ---
print(f"Laster inn data for: {DATASET_NAME}...")
# Bruker load_wine()
dataset = load_wine()
X = dataset.data
y = dataset.target
# Henter funksjonsnavn hvis tilgjengelig
try:
    feature_names = dataset.feature_names
except AttributeError:
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]

n_features = X.shape[1]
print(f"Datasett lastet: {DATASET_NAME} med {n_features} funksjoner.")
print(f"Funksjonsnavn: {feature_names}")

# --- 2. Splitt data EN GANG ---
print("Splitter data i trenings- og testsett...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y # Viktig for klassifikasjon, sikrer proporsjonalitet
)
print(f"Treningssett: {X_train.shape}, Testsett: {X_test.shape}")

# --- 3. Definer modell ---
rf_model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    class_weight='balanced', # Greit å beholde for klassifikasjon
    n_jobs=-1 # Bruk alle tilgjengelige CPU-kjerner for raskere trening
)

# --- 4. Iterer gjennom alle funksjonskombinasjoner ---
results = []
total_combinations = 2**n_features - 1 # Totalt antall ikke-tomme delmengder

# Justerer grensen basert på MAX_COMBINATIONS
limit_combinations = MAX_COMBINATIONS if MAX_COMBINATIONS is not None else total_combinations

print(f"\nStarter evaluering av funksjonsdelmengder for {MODEL_NAME} på {DATASET_NAME}...")
print(f"Valgt feilmetrikk (h_E): 1 - Accuracy")
print(f"Valgt straff (h_P): Epsilon * (Antall funksjoner), med Epsilon = {EPSILON}")
print(f"Total fitness (h): h_E + h_P (Minimeres)")
print(f"Antall funksjoner (N): {n_features}")
print(f"Totalt antall mulige ikke-tomme kombinasjoner: {total_combinations:,}")
if MAX_COMBINATIONS is not None:
    print(f"BEGRENSET til å kjøre maks {limit_combinations:,} kombinasjoner for testing.")
else:
    # Gi et tidsestimat for N=13
    print(f"Kjører alle {limit_combinations:,} kombinasjoner. For N=13 kan dette ta noen minutter til timer.")

start_time = time.time()

# Loopen går nå fra 1 til 2^13 - 1 (eller limit_combinations)
for i in tqdm(range(1, limit_combinations + 1), total=limit_combinations, desc=f"Evaluerer {DATASET_NAME}"):
    # Generer bitmask og finn indekser for valgte funksjoner
    bitmask = bin(i)[2:].zfill(n_features)
    selected_indices = [idx for idx, bit in enumerate(bitmask) if bit == '1']

    if not selected_indices: # Skal ikke skje i praksis når i starter på 1
        continue

    num_selected_features = len(selected_indices)

    # Velg de aktuelle kolonnene fra dataene
    X_train_subset = X_train[:, selected_indices]
    X_test_subset = X_test[:, selected_indices]

    try:
        # Tren modellen
        rf_model.fit(X_train_subset, y_train)

        # Gjør prediksjoner
        y_pred_subset = rf_model.predict(X_test_subset)

        # Beregn ytelse og fitness
        accuracy = accuracy_score(y_test, y_pred_subset)
        h_E = 1.0 - accuracy
        h_P = EPSILON * num_selected_features
        h = h_E + h_P

        # Lagre resultatet
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
        # Logg eventuelle feil
        print(f"\nFEIL under evaluering av kombinasjon {i} (bitmask: {bitmask}): {e}")
        results.append({
            'subset_indices': tuple(selected_indices),
            'subset_bitmask': bitmask,
            'num_features': num_selected_features,
            'accuracy': np.nan, 'error_hE': np.nan, 'penalty_hP': np.nan, 'fitness_h': np.nan
        })


# --- 5. Lagre resultatene ---
print("\nEvaluering fullført (eller stoppet). Konverterer resultater til DataFrame...")
lookup_table_df = pd.DataFrame(results)

# Sorterer tabellen etter fitness (beste først)
# lookup_table_df = lookup_table_df.sort_values(by='fitness_h', ascending=True).reset_index(drop=True)

print(f"Lagrer lookup-tabell til fil: {OUTPUT_FILENAME}")
lookup_table_df.to_csv(OUTPUT_FILENAME, index=False)

end_time = time.time()
total_time = end_time - start_time

print("\n--- Oppsummering ---")
print(f"Datasett: {DATASET_NAME}")
print(f"Modell: {MODEL_NAME}")
print(f"Antall evaluerte kombinasjoner: {len(results)} av {total_combinations}")
print(f"Lookup-tabell lagret som: {OUTPUT_FILENAME}")
print(f"Total tid brukt: {total_time:.2f} sekunder ({total_time/60:.2f} minutter)")
print("\nFørste 5 rader i lookup-tabellen (usortert):")
print(lookup_table_df.head())

# Finner og viser den beste kombinasjonen
if not lookup_table_df['fitness_h'].isnull().all():
    best_row = lookup_table_df.loc[lookup_table_df['fitness_h'].idxmin()]
    print("\nBeste funnet kombinasjon (lavest h):")
    print(f"  Fitness (h): {best_row['fitness_h']:.6f}")
    print(f"  Error (h_E): {best_row['error_hE']:.6f} (Accuracy: {best_row['accuracy']:.6f})")
    print(f"  Penalty (h_P): {best_row['penalty_hP']:.6f}")
    print(f"  Antall funksjoner: {best_row['num_features']}")
    print(f"  Bitmask: {best_row['subset_bitmask']}")
    # For å se hvilke funksjoner:
    # best_indices = best_row['subset_indices']
    # best_feature_names = [feature_names[i] for i in best_indices]
    # print(f"  Funksjonsnavn: {best_feature_names}")
else:
    print("\nIngen gyldige fitness-verdier funnet.")
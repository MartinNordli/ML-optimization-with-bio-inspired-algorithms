import pandas as pd
import numpy as np
# Importer for å hente data fra OpenML
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import time
import warnings

# Ignorerer noen vanlige advarsler
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Konfigurasjon ---
# Oppdatert for Seeds-datasettet
DATASET_NAME = "Seeds"
OPENML_ID = 1499 # Seeds ID på OpenML
MODEL_NAME = "RandomForestClassifier" # Samme modell
EPSILON = 0.01
N_ESTIMATORS = 100
TEST_SIZE = 0.3
RANDOM_STATE = 42
# Oppdatert filnavn
OUTPUT_FILENAME = f"tables/lookup_table_{DATASET_NAME}_{MODEL_NAME}.csv"
# Kjører alle 127 kombinasjoner for N=7
MAX_COMBINATIONS = None

# --- 1. Last inn og forbered data ---
print(f"Laster inn data for: {DATASET_NAME} (OpenML ID: {OPENML_ID})...")
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


    # --- VIKTIG: Forbehandling av y for Seeds ---
    # Klassene er vanligvis '1', '2', '3' -> må bli 0, 1, 2
    unique_targets_raw = np.unique(y_raw)
    print(f"Unike verdier funnet i rå target (y_raw): {unique_targets_raw}")

    if np.all(np.isin(unique_targets_raw, ['1', '2', '3'])):
        print("Mapper target: '1' -> 0, '2' -> 1, '3' -> 2")
        mapper = {'1': 0, '2': 1, '3': 2}
        y = np.array([mapper[val] for val in y_raw])
    elif np.all(np.isin(unique_targets_raw, [1.0, 2.0, 3.0])): # Hvis det er floats
        print("Mapper target: 1.0 -> 0, 2.0 -> 1, 3.0 -> 2")
        mapper = {1.0: 0, 2.0: 1, 3.0: 2}
        y = np.array([mapper[val] for val in y_raw])
    else:
        print(f"ADVARSEL: Ukjent format på target-verdier: {unique_targets_raw}. Prøver direkte konvertering til int.")
        try:
            y = y_raw.astype(int)
            if np.min(y) > 0: # Juster hvis klassene starter på 1
                 print(f"Justerer y ({np.unique(y)}) til å starte på 0.")
                 y = y - np.min(y)
        except Exception as map_e:
             print(f"FEIL under mapping av y: {map_e}. Kan ikke fortsette.")
             exit()

    print(f"Unike verdier i ferdig target (y): {np.unique(y)}")
    # Sjekk at vi har 3 klasser (0, 1, 2)
    if len(np.unique(y)) != 3:
         print(f"ADVARSEL: Forventet 3 klasser (0, 1, 2), men fant {len(np.unique(y))}. Sjekk mappingen.")

    print("Data hentet successfully.")

    # Sjekk for NaN (bør ikke være i Seeds)
    if np.isnan(X).any():
        print("ADVARSEL: NaN (manglende verdier) funnet i X. Krever håndtering.")
        # Implementer imputering her om nødvendig

except Exception as e:
    print(f"FEIL: Kunne ikke hente eller behandle data fra OpenML: {e}")
    print("Sjekk internettforbindelse, OpenML ID (1499), og formatet på dataene.")
    exit()

n_features = X.shape[1]
print(f"Datasett lastet: {DATASET_NAME} med {n_features} funksjoner.")
print(f"Funksjonsnavn: {feature_names}")


# --- 2. Splitt data EN GANG ---
print("Splitter data i trenings- og testsett...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y # Viktig for klassifikasjon med flere klasser
)
print(f"Treningssett: {X_train.shape}, Testsett: {X_test.shape}")

# --- 3. Definer modell ---
rf_model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    class_weight='balanced',
    n_jobs=-1
)

# --- 4. Iterer gjennom alle funksjonskombinasjoner ---
results = []
total_combinations = 2**n_features - 1 # Blir 127 for N=7
limit_combinations = MAX_COMBINATIONS if MAX_COMBINATIONS is not None else total_combinations

print(f"\nStarter evaluering av funksjonsdelmengder for {MODEL_NAME} på {DATASET_NAME}...")
print(f"Antall funksjoner (N): {n_features}")
print(f"Totalt antall mulige ikke-tomme kombinasjoner: {total_combinations:,}")
if MAX_COMBINATIONS is not None:
    print(f"BEGRENSET til å kjøre maks {limit_combinations:,} kombinasjoner.")
else:
    print(f"Kjører alle {limit_combinations:,} kombinasjoner (N=7). Dette går raskt!")

start_time = time.time()

# Løkken itererer nå bare 127 ganger
for i in tqdm(range(1, limit_combinations + 1), total=limit_combinations, desc=f"Evaluerer {DATASET_NAME}"):
    bitmask = bin(i)[2:].zfill(n_features)
    selected_indices = [idx for idx, bit in enumerate(bitmask) if bit == '1']

    if not selected_indices: continue

    num_selected_features = len(selected_indices)
    X_train_subset = X_train[:, selected_indices]
    X_test_subset = X_test[:, selected_indices]

    try:
        # Tren, prediker, evaluer
        rf_model.fit(X_train_subset, y_train)
        y_pred_subset = rf_model.predict(X_test_subset)
        accuracy = accuracy_score(y_test, y_pred_subset)
        h_E = 1.0 - accuracy
        h_P = EPSILON * num_selected_features
        h = h_E + h_P

        # Lagre resultat
        results.append({
            'subset_indices': tuple(selected_indices), 'subset_bitmask': bitmask,
            'num_features': num_selected_features, 'accuracy': accuracy,
            'error_hE': h_E, 'penalty_hP': h_P, 'fitness_h': h
        })

    except Exception as e:
        print(f"\nFEIL under evaluering av kombinasjon {i} (bitmask: {bitmask}): {e}")
        results.append({
             'subset_indices': tuple(selected_indices), 'subset_bitmask': bitmask,
             'num_features': num_selected_features, 'accuracy': np.nan,
             'error_hE': np.nan, 'penalty_hP': np.nan, 'fitness_h': np.nan
        })

# --- 5. Lagre resultatene ---
print("\nEvaluering fullført. Konverterer resultater til DataFrame...")
lookup_table_df = pd.DataFrame(results)

print(f"Lagrer lookup-tabell til fil: {OUTPUT_FILENAME}")
lookup_table_df.to_csv(OUTPUT_FILENAME, index=False)

end_time = time.time()
total_time = end_time - start_time

print("\n--- Oppsummering ---")
print(f"Datasett: {DATASET_NAME}")
print(f"Modell: {MODEL_NAME}")
print(f"Antall evaluerte kombinasjoner: {len(results)} av {total_combinations}")
print(f"Lookup-tabell lagret som: {OUTPUT_FILENAME}")
print(f"Total tid brukt: {total_time:.2f} sekunder") # Bør være veldig raskt
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
    # try:
    #     best_feature_names = [feature_names[i] for i in best_indices]
    #     print(f"  Funksjonsnavn: {best_feature_names}")
    # except NameError:
    #     print(f"  Funksjonsindekser: {best_indices}")

else:
    print("\nIngen gyldige fitness-verdier funnet.")
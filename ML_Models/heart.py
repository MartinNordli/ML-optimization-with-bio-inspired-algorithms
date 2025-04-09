import pandas as pd
import numpy as np
# Importer for å hente data fra OpenML
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Imputering hvis nødvendig (vanligvis ikke for Statlog Heart)
# from sklearn.impute import SimpleImputer
from tqdm import tqdm
import time
import warnings

# Ignorerer noen vanlige advarsler fra sklearn/openml
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# --- Konfigurasjon ---
# Oppdatert for Heart Disease (Statlog) datasettet
DATASET_NAME = "Heart_Disease_Statlog"
OPENML_ID = 53 # Statlog (Heart) ID på OpenML
MODEL_NAME = "RandomForestClassifier" # Samme modell
EPSILON = 0.01
N_ESTIMATORS = 100
TEST_SIZE = 0.3
RANDOM_STATE = 42
# Oppdatert filnavn
OUTPUT_FILENAME = f"tables/lookup_table_{DATASET_NAME}_{MODEL_NAME}.csv"
# Kjører alle 8191 kombinasjoner for N=13
MAX_COMBINATIONS = None
# MAX_COMBINATIONS = 100 # For rask test

# --- 1. Last inn og forbered data ---
print(f"Laster inn data for: {DATASET_NAME} (OpenML ID: {OPENML_ID})...")
X = None
y = None
feature_names = None
try:
    # parser='liac-arff' er ofte nødvendig for ARFF filer fra OpenML
    heart_data = fetch_openml(data_id=OPENML_ID, as_frame=False, parser='liac-arff')
    X = heart_data.data
    y_raw = heart_data.target
    try:
      feature_names = heart_data.feature_names
    except AttributeError:
      feature_names = [f'feature_{i}' for i in range(X.shape[1])]


    # --- VIKTIG: Forbehandling av y ---
    # Sjekk rå-verdiene i target-variabelen
    unique_targets_raw = np.unique(y_raw)
    print(f"Unike verdier funnet i rå target (y_raw): {unique_targets_raw}")

    # Antar at '1' betyr "fravær" (klasse 0) og '2' betyr "nærvær" (klasse 1)
    # Dette er vanlig for Statlog Heart, men VERIFISER basert på output over!
    if np.all(np.isin(unique_targets_raw, ['1', '2'])):
        print("Mapper target: '1' -> 0, '2' -> 1")
        y = np.array([0 if val == '1' else 1 for val in y_raw])
    elif np.all(np.isin(unique_targets_raw, [1.0, 2.0])): # Hvis det er floats
         print("Mapper target: 1.0 -> 0, 2.0 -> 1")
         y = np.array([0 if val == 1.0 else 1 for val in y_raw])
    elif np.all(np.isin(unique_targets_raw, ['absent', 'present'])):
        print("Mapper target: 'absent' -> 0, 'present' -> 1")
        y = np.array([0 if val == 'absent' else 1 for val in y_raw])
    else:
        print(f"ADVARSEL: Ukjent format på target-verdier: {unique_targets_raw}. Prøver direkte konvertering til int.")
        # Fallback - kan kreve manuell justering hvis dette feiler
        y = y_raw.astype(int)
        # Hvis dette var f.eks. 0 og 1 originalt, trengs ingen mapping.

    print(f"Unike verdier i ferdig target (y): {np.unique(y)}")
    print("Data hentet successfully.")

    # --- NYTT: Undersøk funksjonstyper ---
    print("\n--- Undersøker funksjonstyper ---")
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])] # Fallback hvis navn ikke finnes

    print(f"Datatyper i X (numpy array): {X.dtype}")
    print("Antall unike verdier per funksjon:")
    for i, name in enumerate(feature_names):
        unique_values = np.unique(X[:, i])
        num_unique = len(unique_values)
        # Viser de første par unike verdiene for kontekst, spesielt for kategoriske
        display_values = unique_values[:5] if num_unique > 5 else unique_values
        print(f"  {i}: {name} - {num_unique} unike verdier. Eksempler: {display_values}")
        # Heuristikk for å foreslå kategoriske (kan justeres)
        if num_unique <= 10: # Anta kategorisk hvis 10 eller færre unike verdier
            print(f"     -> Foreslått type: Kategorisk")
        else:
            print(f"     -> Foreslått type: Numerisk")
    print("----------------------------------\n")

    # Sjekk for NaN-verdier (bør ikke være i Statlog Heart)
    if np.isnan(X).any():
        print("ADVARSEL: NaN (manglende verdier) funnet i X. Dette krever håndtering (f.eks. imputering).")
        # Implementer imputering her om nødvendig
        # imputer = SimpleImputer(strategy='mean')
        # X = imputer.fit_transform(X)

except Exception as e:
    print(f"FEIL: Kunne ikke hente eller behandle data fra OpenML: {e}")
    print("Sjekk internettforbindelse, OpenML ID, og formatet på dataene.")
    exit() # Avslutt hvis data ikke kan lastes

n_features = X.shape[1]
print(f"Datasett lastet: {DATASET_NAME} med {n_features} funksjoner.")
print(f"Funksjonsnavn: {feature_names}")


# --- 2. Splitt data EN GANG ---
print("Splitter data i trenings- og testsett...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y # Viktig for klassifikasjon
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
total_combinations = 2**n_features - 1
limit_combinations = MAX_COMBINATIONS if MAX_COMBINATIONS is not None else total_combinations

print(f"\nStarter evaluering av funksjonsdelmengder for {MODEL_NAME} på {DATASET_NAME}...")
# ... (resten av print-utsagnene er like som før) ...
print(f"Antall funksjoner (N): {n_features}")
print(f"Totalt antall mulige ikke-tomme kombinasjoner: {total_combinations:,}")
if MAX_COMBINATIONS is not None:
    print(f"BEGRENSET til å kjøre maks {limit_combinations:,} kombinasjoner.")
else:
    print(f"Kjører alle {limit_combinations:,} kombinasjoner (N=13).")


start_time = time.time()

# Løkken er identisk, itererer over 2^13 - 1 kombinasjoner
for i in tqdm(range(1, limit_combinations + 1), total=limit_combinations, desc=f"Evaluerer {DATASET_NAME}"):
    bitmask = bin(i)[2:].zfill(n_features)
    selected_indices = [idx for idx, bit in enumerate(bitmask) if bit == '1']

    if not selected_indices: continue

    num_selected_features = len(selected_indices)
    X_train_subset = X_train[:, selected_indices]
    X_test_subset = X_test[:, selected_indices]

    try:
        # Tren, prediker, evaluer (samme som før)
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
print("\nEvaluering fullført (eller stoppet). Konverterer resultater til DataFrame...")
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
# run_all.py (ENDELIG KORRIGERT versjon)

import pandas as pd
import numpy as np
import time
import math # Importer math for math.isfinite
import os

# --- Importer dine algoritme-funksjoner ---
try:
    # Anta at funksjonene nå returnerer som spesifisert under!
    from simple_ga import run_simple_ga       # FORVENTER dict retur nå: {'best_bitstring':..., 'best_fitness':..., 'history':...}
    from swarm import run_bpso             # FORVENTER tuple retur: (gbest_string, gbest_fit, history)
    from multi_objective import run_nsga2    # FORVENTER dict retur nå: {'pareto_solutions':..., 'pareto_objectives':...}
    print("Algoritme-funksjoner importert.")
except ImportError as e:
    print(f"FEIL: Kunne ikke importere algoritme-funksjon(er). Sjekk filnavn og funksjonsdefinisjoner.")
    print(f"Feilmelding: {e}")
    exit()
except Exception as import_e:
     print(f"En annen feil oppstod under import: {import_e}")
     exit()


# --- Konfigurasjon ---
ALGORITHMS = ['GA', 'BPSO', 'NSGA-II']
DATASETS = ['Wine', 'Heart_Disease_Statlog', 'Seeds']
N_RUNS = 10
RESULTS_FOLDER = 'results'
RESULTS_FILE = os.path.join(RESULTS_FOLDER, 'experiment_results.csv')
TABLES_FOLDER = 'tables'
MODEL_NAME = "RandomForestClassifier"

# Lag resultat-mappe
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Kjente globale optima (fitness) for GA/BPSO
known_global_optima_fitness = {
    'Wine': 0.030000,
    'Heart_Disease_Statlog': 0.148765,
    'Seeds': 0.051746
}

# --- Hjelpefunksjon for datalasting ---
def load_data_for_dataset(dataset_name):
    """Laster lookup CSV og returnerer n_features, fitness_dict, objectives_dict."""
    lookup_file = os.path.join(TABLES_FOLDER, f"lookup_table_{dataset_name}_{MODEL_NAME}.csv")
    print(f"  Laster data fra: {lookup_file}")
    if not os.path.exists(lookup_file):
         raise FileNotFoundError(f"Lookup-fil ikke funnet: {lookup_file}")

    df = pd.read_csv(lookup_file)
    required_cols = ['subset_bitmask', 'num_features', 'error_hE', 'fitness_h']
    if not all(col in df.columns for col in required_cols):
         raise ValueError(f"Nødvendige kolonner mangler i {lookup_file}.")

    df['subset_bitmask'] = df['subset_bitmask'].astype(str)
    n_features = int(df['num_features'].max())
    df['subset_bitmask'] = df['subset_bitmask'].str.zfill(n_features)
    df.dropna(subset=['error_hE', 'num_features', 'fitness_h'], inplace=True)
    if df.empty:
        raise ValueError(f"Ingen gyldige data i {lookup_file} etter fjerning av NaN.")

    fitness_dict = pd.Series(df.fitness_h.values, index=df.subset_bitmask).to_dict()
    objectives_dict = {row['subset_bitmask']: (row['error_hE'], int(row['num_features']))
                       for _, row in df.iterrows()}

    print(f"  Data lastet: N={n_features}, Antall løsninger: {len(objectives_dict)}")
    return n_features, fitness_dict, objectives_dict

# --- Hovedeksperimentløkke ---
all_run_results = []

for dataset_name in DATASETS:
    print(f"\n{'='*10} Behandler Datasett: {dataset_name} {'='*10}")

    try:
        n_features, fitness_dict, objectives_dict = load_data_for_dataset(dataset_name)
    except Exception as e:
        print(f"  FEIL ved lasting av data for {dataset_name}: {e}. Hopper over.")
        continue

    for algo_name in ALGORITHMS:
        print(f"\n  --- Kjører Algoritme: {algo_name} ---")
        for run in range(1, N_RUNS + 1):
            print(f"    Kjøring {run}/{N_RUNS}...")
            start_run_time = time.time()

            result = {
                'Algorithm': algo_name, 'Dataset': dataset_name, 'Run': run,
                'N_Features_Total': n_features, 'Best_Fitness': np.nan,
                'Found_Global_Opt_Fitness': False, 'Best_Bitstring': None,
                'Pareto_Front_Size': np.nan, 'Min_Error_on_Front': np.nan,
                'Min_Feats_on_Front': np.nan, 'Execution_Time': np.nan
            }

            try:
                if algo_name == 'GA':
                    # --- KORRIGERT HÅNDTERING FOR GA ---
                    # Anta run_simple_ga returnerer DICT nå (basert på siste korreksjon av simple_ga.py)
                    ga_result = run_simple_ga(fitness_dict, n_features)
                    current_best_fitness = ga_result.get('best_fitness', np.nan)
                    result['Best_Fitness'] = current_best_fitness
                    result['Best_Bitstring'] = ga_result.get('best_bitstring', None)
                    # Robust sjekk for isfinite
                    is_valid_finite = isinstance(current_best_fitness, (int, float)) and math.isfinite(current_best_fitness)
                    if is_valid_finite:
                        result['Found_Global_Opt_Fitness'] = np.isclose(current_best_fitness, known_global_optima_fitness[dataset_name])
                    else:
                        result['Found_Global_Opt_Fitness'] = False

                elif algo_name == 'BPSO':
                     # --- KORRIGERT HÅNDTERING FOR BPSO ---
                    # Anta run_bpso returnerer TUPLE: (gbest_string, gbest_fit, history)
                    gbest_string, gbest_fit, history = run_bpso(fitness_dict, n_features)
                    result['Best_Fitness'] = gbest_fit
                    result['Best_Bitstring'] = gbest_string
                    # Robust sjekk for isfinite
                    is_valid_finite = isinstance(gbest_fit, (int, float)) and math.isfinite(gbest_fit)
                    if is_valid_finite:
                        result['Found_Global_Opt_Fitness'] = np.isclose(gbest_fit, known_global_optima_fitness[dataset_name])
                    else:
                        result['Found_Global_Opt_Fitness'] = False

                elif algo_name == 'NSGA-II':
                    # Anta run_nsga2 returnerer DICT nå: {'pareto_solutions':..., 'pareto_objectives':...}
                    nsga2_result = run_nsga2(objectives_dict, n_features)
                    objectives = nsga2_result.get('pareto_objectives', [])
                    solutions = nsga2_result.get('pareto_solutions', [])
                    if objectives:
                        result['Pareto_Front_Size'] = len(solutions)
                        try:
                            valid_errors = [obj[0] for obj in objectives if np.isfinite(obj[0])]
                            valid_feats = [obj[1] for obj in objectives if np.isfinite(obj[1])]
                            result['Min_Error_on_Front'] = min(valid_errors) if valid_errors else np.nan
                            result['Min_Feats_on_Front'] = min(valid_feats) if valid_feats else np.nan
                        except ValueError: pass
                    else:
                        result['Pareto_Front_Size'] = 0

            except Exception as run_e:
                print(f"      FEIL under kjøring {run} av {algo_name} på {dataset_name}: {run_e}")

            end_run_time = time.time()
            result['Execution_Time'] = end_run_time - start_run_time
            all_run_results.append(result)
            print(f"      Fullført på {result['Execution_Time']:.2f} sek.")

# --- Lagre alle resultater ---
print("\nAlle eksperimenter fullført.")
if all_run_results:
    results_df = pd.DataFrame(all_run_results)
    column_order = [
        'Algorithm', 'Dataset', 'Run', 'N_Features_Total', 'Best_Fitness',
        'Found_Global_Opt_Fitness', 'Best_Bitstring', 'Pareto_Front_Size',
        'Min_Error_on_Front', 'Min_Feats_on_Front', 'Execution_Time'
    ]
    results_df = results_df.reindex(columns=column_order)

    try:
        results_df.to_csv(RESULTS_FILE, index=False)
        print(f"Resultatene er lagret i: {RESULTS_FILE}")
    except Exception as save_e:
        print(f"Kunne ikke lagre resultater til CSV: {save_e}")

    # --- Enkel Analyse Eksempel ---
    print("\n--- Oppsummering (Gjennomsnitt/Statistikk per Alg/Datasett) ---")
    results_df_agg = results_df.dropna(axis=1, how='all')
    if 'Found_Global_Opt_Fitness' in results_df_agg.columns:
         results_df_agg['Found_Global_Opt_Fitness'] = results_df_agg['Found_Global_Opt_Fitness'].fillna(0).astype(float)

    agg_funcs = {
        'Run': 'count',
        'Best_Fitness': ['mean', 'std'],
        'Found_Global_Opt_Fitness': 'mean',
        'Pareto_Front_Size': ['mean', 'std'],
        'Min_Error_on_Front': 'mean',
        'Min_Feats_on_Front': 'mean',
        'Execution_Time': ['mean', 'std']
    }
    agg_funcs = {k: v for k, v in agg_funcs.items() if k in results_df_agg.columns}

    try:
        summary = results_df_agg.groupby(['Algorithm', 'Dataset']).agg(agg_funcs)
        summary.columns = ['_'.join(col).strip() if isinstance(col, tuple) and col[1]!='' else col[0] for col in summary.columns.values]
        summary = summary.rename(columns={'Run_count': 'Num_Runs',
                                          'Found_Global_Opt_Fitness_mean': 'Success_Rate_Global_Fit'})
        print(summary.round(4))
    except Exception as agg_e:
        print(f"Kunne ikke beregne oppsummering: {agg_e}")
        print("Viser rådataens kolonner og typer:")
        print(results_df.info())

else:
    print("Ingen resultater ble generert.")

print("\nEksperimentering fullført.")
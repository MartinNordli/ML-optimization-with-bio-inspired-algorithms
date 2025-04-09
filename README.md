# Project: Understanding Optimization Landscapes using Bio-inspired Algorithms

## Overview

This project implements and compares different bio-inspired optimization algorithms for the task of feature selection in machine learning. The core idea is to:

1.  **Create Surrogate Fitness Landscapes:** Instead of repeatedly training a machine learning model during optimization (which is computationally expensive), we first pre-compute the performance (and complexity) for *all* possible feature subsets for selected datasets. This creates lookup tables that act as fast surrogate models of the fitness landscape.
2.  **Implement Bio-inspired Algorithms:** Implement three types of algorithms:
    * A single-objective Genetic Algorithm (GA).
    * A Swarm Intelligence algorithm (Binary Particle Swarm Optimization - BPSO).
    * A Multi-objective Evolutionary Algorithm (NSGA-II).
3.  **Analyze Algorithm Performance:** Run these algorithms using the lookup tables as their fitness function to study how they navigate different landscapes and compare their effectiveness in finding optimal or near-optimal feature subsets.

The project focuses on understanding the interplay between algorithm behavior and problem structure (the landscape), illustrating concepts like multi-modality and the No-Free-Lunch theorem in optimization.

## Dependencies

This project requires Python 3 and the following libraries:

* `numpy`: For numerical operations and array handling.
* `pandas`: For data manipulation and reading/writing CSV files (lookup tables, results).
* `matplotlib`: For plotting fitness history and Pareto fronts (used in standalone testing and potentially `run_all.py`).
* `scikit-learn`: Used for loading datasets (like Wine) and potentially needed if regenerating lookup tables.
* `tqdm`: (Optional, but used in some scripts) For displaying progress bars.

## Results

The main output of the experimentation is the results/experiment_results.csv file. This file contains detailed performance metrics for each algorithm run across the different datasets. The summary table printed at the end of run_all.py provides aggregated statistics (mean, standard deviation, success rate) useful for comparing the algorithms' consistency and effectiveness on the different fitness landscapes. Analysis of these results forms the basis for the project report/presentation.

## Acknowledgements

    Datasets sourced from scikit-learn (load_wine) and OpenML/UCI ML Repository (Heart Disease - Statlog ID 53, Seeds ID 1499).
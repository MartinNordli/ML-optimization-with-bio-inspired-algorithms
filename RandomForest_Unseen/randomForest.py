#!/usr/bin/env python3
"""
randomForest.py

Binary Particle Swarm Optimisation (BPSO) wrapper around a Random‑Forest feature‑subset
fitness function, as specified in Project 3 – Step 6 (Updated).

Usage
-----
$ python randomForest.py --task 1  # Heart Disease (Cleveland)
$ python randomForest.py --task 2  # Zoo
$ python randomForest.py --task 3  # Letter Recognition

All other settings (data paths, swarm hyper‑parameters, RF seeds, etc.) are defined
inside the main() function and can be tweaked directly in this file.
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)


def _prepare_dataset(task_id: int):
    """Return (X, y) DataFrames for the given task, numeric‑only, NaNs filled."""
    ds_map = {1: 45, 2: 111, 3: 59}
    if task_id not in ds_map:
        raise ValueError("task must be 1, 2 or 3")
    ds = fetch_ucirepo(id=ds_map[task_id])
    X = ds.data.features.copy().apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    y = ds.data.targets.copy()
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    if y.dtype == object:
        y = pd.factorize(y)[0]
    return X.reset_index(drop=True), y


class RandomForestFitness:
    """Callable that returns (h(x), accuracy) for a bit‑string selection."""

    _EPSILON = {1: 0.0, 2: 1 / 64, 3: 1 / 8}
    _GLOBAL_OPT = {
        1: np.array([1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0], dtype=int),
        2: np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0], dtype=int),
        3: np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0], dtype=int),
    }

    def __init__(self, task_id: int, tables_dir: Path):
        self.task_id = task_id
        self.tables_dir = tables_dir
        self.epsilon = self._EPSILON[task_id]
        self.X, self.y = _prepare_dataset(task_id)
        self.d = self.X.shape[1]
        self.X_tr, self.X_te, self.y_tr, self.y_te = train_test_split(
            self.X, self.y, test_size=0.3, random_state=123, stratify=self.y
        )
        self.global_opt = self._GLOBAL_OPT[task_id]
        self.h5_path = self._find_table()

    # ------------------------------------------------------------------ helpers
    def _find_table(self):
        table_name = {1: "5-heart-c_rf_mat.h5", 2: "8-zoo_rf_mat.h5"}.get(self.task_id)
        if table_name is None:
            return None
        p = self.tables_dir / table_name
        print(f"DEBUG: Checking for table at path: {p}") # <-- Add this line
        print(f"DEBUG: Does path exist? {p.exists()}") # <-- Add this line
        return p if p.exists() else None

    def _lookup_accuracy(self, decimal_index: int):
        with h5py.File(self.h5_path, "r") as h5f:
            data = h5f["data"]
            # print(f"DEBUG: Shape of 'data' dataset read by h5py: {data.shape}") # <-- Add this

            return float(data[0, decimal_index - 1])

    def _train_accuracy(self, cols):
        if cols is None:
            return 0.0  # empty subset ⇒ no predictive power
        accs = []
        for i in range(30):
            rf = RandomForestClassifier(
                n_estimators=30,
                criterion="gini",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features=None,
                bootstrap=True,
                random_state=456, # 456 + i,
                n_jobs=-1,
            )
            rf.fit(self.X_tr.iloc[:, cols], self.y_tr)
            preds = rf.predict(self.X_te.iloc[:, cols])
            accs.append(accuracy_score(self.y_te, preds))
        return float(np.mean(accs))

    # ------------------------------------------------------------------- public
    def __call__(self, bits):
        """Evaluate one bit‑string, return (h(x), accuracy)."""
        bits = np.asarray(bits, dtype=int)
        if bits.size != self.d:
            raise ValueError("bit-string length does not match #features")

        # --- decimal index (LSB = right‑most bit) follows project spec
        decimal_idx = int(np.dot(bits[::-1], 1 << np.arange(self.d)))

        # --- accuracy: use lookup table if available, else train forests
        if self.h5_path is not None and decimal_idx > 0:
            acc = self._lookup_accuracy(decimal_idx)
        else:
            # identity mapping: bit i picks column i (0‑based)
            cols = [i for i, b in enumerate(bits) if b]
            acc = self._train_accuracy(cols if cols else None)

        error = 1.0 - acc
        penalty = self.epsilon * bits.sum()
        return error + penalty, acc

    # convenience for reporting
    def hamming_distance(self, bits):
        return int(np.sum(bits != self.global_opt))


# ---------------------------------------------------------------- BPSO core

def bpso(
    fitness,
    dim: int,
    swarm_size: int = 40,
    iters: int = 100,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    v_max: float = 4.0,
):
    rng = np.random.default_rng(456)
    X = rng.integers(0, 2, size=(swarm_size, dim))
    V = rng.uniform(-v_max, v_max, size=(swarm_size, dim))

    pbest = X.copy()
    pbest_val = np.array([fitness(x)[0] for x in X])
    gbest_idx = int(np.argmin(pbest_val))
    gbest = pbest[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]

    history = [gbest_val]
    for t in range(iters):
        r1, r2 = rng.random((2, swarm_size, dim))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        V = np.clip(V, -v_max, v_max)
        S = 1.0 / (1.0 + np.exp(-V))
        X = (rng.random((swarm_size, dim)) < S).astype(int)

        for i in range(swarm_size):
            val, _ = fitness(X[i])
            if val < pbest_val[i]:
                pbest[i], pbest_val[i] = X[i].copy(), val
                if val < gbest_val:
                    gbest, gbest_val = X[i].copy(), val
        history.append(gbest_val)
        print(f"Iter {t + 1:3d} / {iters} | best h(x) = {gbest_val:.4f}")

    return gbest, gbest_val, history


# ------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(
        description="Binary PSO + Random‑Forest feature selector (Project 3 – Step 6)"
    )
    parser.add_argument("--task", required=True, type=int, help="Task 1, 2 or 3")
    args = parser.parse_args()

    tables_dir = Path(__file__).resolve().parent.parent / "unseen_tables"
    fitness = RandomForestFitness(args.task, tables_dir)

    # ---- swarm & RF settings (edit as desired) -------------------
    swarm_size = 40
    iters = 100
    w, c1, c2, v_max = 0.7, 1.5, 1.5, 4.0
    # --------------------------------------------------------------

    start = time.time()
    best_bits, best_val, hist = bpso(
        fitness,
        dim=fitness.d,
        swarm_size=swarm_size,
        iters=iters,
        w=w,
        c1=c1,
        c2=c2,
        v_max=v_max,
    )
    elapsed = time.time() - start

    h_val, acc = fitness(best_bits)
    ham = fitness.hamming_distance(best_bits)

    print("\n===== run summary =====")
    print("task", args.task)
    print("bit‑string (left→right columns):", best_bits.tolist())
    print("Hamming distance to global optimum:", ham)
    print(f"mean RF accuracy: {acc:.4f}")
    print(f"h(x) = {h_val:.4f}")
    print(f"elapsed: {elapsed:.1f}s")

    out = {
        "task": args.task,
        "bitstring": best_bits.tolist(),
        "hamming_distance": ham,
        "accuracy": acc,
        "h": h_val,
        "history": hist,
    }
    out_path = Path(f"task{args.task}_run.json")
    with open(out_path, "w") as fp:
        json.dump(out, fp, indent=2)
    print("results written to", out_path)


if __name__ == "__main__":
    main()

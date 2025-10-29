#!/usr/bin/env python3

import pickle
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def main():
    if not Path("data.txt").exists():
        raise SystemExit("data.txt not found. Run prepare_data.py first.")
    if not Path("class_names.npy").exists():
        raise SystemExit("class_names.npy not found. Run prepare_data.py first.")

    class_names = np.load("class_names.npy", allow_pickle=True).tolist()
    data = np.loadtxt("data.txt")
    X, y = data[:, :-1], data[:, -1].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("pca", PCA(n_components=0.95, svd_solver="full")),  # keep 95% variance
        ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42))
    ])

    param_grid = {
        "clf__C": [0.5, 1, 2, 5, 10],
        "clf__gamma": ["scale", 0.02, 0.05, 0.1]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    gs = GridSearchCV(
        pipe, param_grid, scoring="accuracy", cv=cv, n_jobs=-1, verbose=0
    )
    gs.fit(X_tr, y_tr)

    print("Best params:", gs.best_params_)
    print("CV best score (mean):", gs.best_score_)

    y_pred = gs.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    print("\n== Holdout Test ==")
    print("Accuracy:", f"{acc:.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_te, y_pred))
    print("\nClassification report:")
    print(classification_report(y_te, y_pred, target_names=[str(c).upper() for c in class_names], digits=3))

    with open("model", "wb") as f:
        pickle.dump(gs.best_estimator_, f)
    print("\nSaved model -> ./model (pickle pipeline)")

if __name__ == "__main__":
    main()

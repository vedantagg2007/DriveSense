"""KNN tuning with preprocessing and oversampling confined to training folds."""

from collections import Counter

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def fit_condition_model(X, y):
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    if X.ndim != 2 or len(X) != len(y) or len(y) == 0:
        raise ValueError("Provide a nonempty feature matrix and matching labels.")
    counts = Counter(y)
    if len(counts) < 2 or min(counts.values()) < 4:
        raise ValueError("Need at least two classes with four valid segments each.")
    # Split original segments first. Validation is neither scaled nor oversampled
    # until the fitted pipeline transforms it during prediction.
    test_count = max(len(counts), int(np.ceil(len(y) * 0.2)))
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_count, random_state=42, stratify=y
    )
    folds = min(5, min(Counter(y_train).values()))
    if folds < 2:
        raise ValueError("Insufficient training segments for cross-validation.")
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("balance", RandomOverSampler(random_state=42)),
        ("knn", KNeighborsClassifier()),
    ])
    # Conservative bound: every candidate fits even the smallest training fold.
    min_fold_train = len(y_train) - int(np.ceil(len(y_train) / folds))
    search = GridSearchCV(
        pipeline,
        {"knn__n_neighbors": list(range(1, min(20, min_fold_train) + 1))},
        cv=StratifiedKFold(folds, shuffle=True, random_state=42),
        scoring="accuracy", error_score="raise", n_jobs=1,
    )
    search.fit(X_train, y_train)
    return (search.best_estimator_, X_val, y_val,
            search.best_params_["knn__n_neighbors"], float(search.best_score_))

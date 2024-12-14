#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 11:36:17 2024

@author: tommasofogarin
"""
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
"""
La grid search può portare alla selezione di modelli troppo complessi, che tendono a sovra-adattarsi ai dati di training.
"""
# Caricamento del dataset
columns = [f"feature_{i+1}" for i in range(57)] + ["label"]
data = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data",
    header=None,
    names=columns,
)

# Bilanciamento delle classi
spam = data[data["label"] == 1]
non_spam_sample = data[data["label"] == 0].sample(n=len(spam), random_state=42)
data_balanced = pd.concat([spam, non_spam_sample]).sample(frac=1, random_state=42)

# Separazione caratteristiche ed etichette
X = data_balanced.iloc[:, :54]  # Prime 54 caratteristiche (frequenze di parole/simboli)
y = data_balanced.iloc[:, 57]  # Etichette (spam/ham)

# Applicazione trasformazione TF-IDF
tfidf = TfidfTransformer()
X_tfidf = tfidf.fit_transform(X).toarray()

# Definizione dei modelli e dei parametri di ricerca
models_with_params = {
    "Linear SVM": (SVC(kernel="linear", random_state=42), {"C": [0.1, 1, 10]}),
    "Polynomial SVM (deg 2)": (SVC(kernel="poly", degree=2, random_state=42), {"C": [0.1, 1, 10]}),
    "RBF SVM": (SVC(kernel="rbf", random_state=42), {"C": [0.1, 1, 10], "gamma": [0.1, 1, 10]}),
    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {"n_estimators": [50, 100, 200]},
    ),
}

# Esecuzione della Grid Search
for model_name, (model, param_grid) in models_with_params.items():
    print(f"\nEseguendo Grid Search per: {model_name}")
    grid_search = GridSearchCV(model, param_grid, cv=10, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_tfidf, y)

    print(f"Migliori parametri per {model_name}: {grid_search.best_params_}")
    print(f"Miglior score di cross-validation per {model_name}: {grid_search.best_score_:.4f}")
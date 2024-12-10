#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:10:34 2024

@author: tommasofogarin e manuelmartini
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate

from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer

# Carica il dataset
columns = [f"feature_{i+1}" for i in range(57)] + ["label"]
data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data", header=None, names=columns)
print(data)

# Separiamo le caratteristiche dalle etichette
y = data.iloc[:, 57]   # Usa la 58esima caratteristica come etichetta (spam/ham)
X = data.iloc[:, :54]  # Usa le prime 54 caratteristiche (frequenze di parole/simboli)

# Bilanciamento delle classi sapendo che #(spam) < #(non_spam)
spam = X[X["label"] == 1]
non_spam_sample = X[X["label"] == 0].sample(n=len(spam), random_state=42)
X_bal = pd.concat([spam, non_spam_sample]).sample(frac=1, random_state=42)  # Mischiare le righe

# Applicare la trasformazione TF-IDF
tfidf = TfidfTransformer()
X_tfidf = tfidf.fit_transform(X_bal).toarray()

# Standardizzare i dati
#scaler = StandardScaler()
#X_tfidf = scaler.fit_transform(X_tfidf)

# Train-test split
#X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.1, random_state=42)


# Modelli con parametri base
models = {
    "Linear Kernel": SVC(kernel='linear', C=1),
    "Poly Kernel (deg 2)": SVC(kernel='poly', degree=2, C=1),
    "RBF Kernel": SVC(kernel='rbf', gamma='scale', C=1),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GaussianNB": GaussianNB(),
    "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5)
}

# Funzione per valutare un modello con cross-validation
def evaluate_model(model, X_tfidf, y):
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, average='weighted'),
        'recall': make_scorer(recall_score, average='weighted'),
        'f1': make_scorer(f1_score, average='weighted')
    }

    # Cross-validation con misurazione dei tempi e metriche multiple
    cv_results = cross_validate(
        model,
        X_tfidf,  # Dati completi
        y,
        cv=KFold(n_splits=10, shuffle=True, random_state=42),
        scoring=scoring,
        return_train_score=False,  # Solo metriche di test
        n_jobs=-1  # Parallelizza per velocizzare
    )

    # Calcolo statistiche
    mean_acc = np.mean(cv_results['test_accuracy'])
    std_acc = np.std(cv_results['test_accuracy'])
    mean_precision = np.mean(cv_results['test_precision'])
    mean_recall = np.mean(cv_results['test_recall'])
    mean_f1 = np.mean(cv_results['test_f1'])
    mean_train_time = np.mean(cv_results['fit_time'])
    mean_predict_time = np.mean(cv_results['score_time'])

    # Stampa dei risultati
    print(f"Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Precision: {mean_precision:.4f}")
    print(f"Recall: {mean_recall:.4f}")
    print(f"F1-Score: {mean_f1:.4f}")
    print(f"Training time: {mean_train_time:.4f} seconds")
    print(f"Prediction time: {mean_predict_time:.4f} seconds")

    # Salvataggio risultati
    accuracies.append(mean_acc)
    std_accuracies.append(std_acc)
    train_times.append(mean_train_time)
    predict_times.append(mean_predict_time)
    model_names.append(name)

# Liste per i risultati
accuracies = []
std_accuracies = []
train_times = []
predict_times = []
model_names = []

# Valutazione di ogni modello
for name, model in models.items():
    print(f"\nTraining and Evaluating {name}")
    evaluate_model(model, X_tfidf, y)

# Grafico delle accuratezze con deviazione standard
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(model_names, accuracies, xerr=std_accuracies, capsize=5, color='skyblue')
ax.set_xlabel("Accuracy")
ax.set_title("Model Comparison - Accuracy")

for i, v in enumerate(accuracies):
    ax.text(v + 0.01, i, f"{v:.3f}", va='center')

plt.tight_layout()
plt.show()

# Grafico separato per i tempi
fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.35
index = np.arange(len(model_names))

# Stacked bars per tempi di training e prediction
bar1 = ax.barh(index, train_times, bar_width, color='lightgreen', label='Training Time')
bar2 = ax.barh(index, predict_times, bar_width, left=train_times, color='lightcoral', label='Prediction Time')

ax.set_xlabel("Time (seconds)")
ax.set_title("Model Comparison - Time")
ax.set_yticks(index)
ax.set_yticklabels(model_names)
ax.legend()

# Annotazioni dei tempi
for i, (train_time, predict_time) in enumerate(zip(train_times, predict_times)):
    ax.text(train_time + predict_time + 0.01, i, f"{train_time + predict_time:.3f}", va='center')

plt.tight_layout()
plt.show()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 16:23:10 2024

@author: tommasofogarin
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfTransformer

# Caricamento dataset
columns = [f"feature_{i+1}" for i in range(57)] + ["label"]
data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data", header=None, names=columns)

# Bilanciamento delle classi
spam = data[data["label"] == 1]
non_spam_sample = data[data["label"] == 0].sample(n=len(spam), random_state=42)
data_bal = pd.concat([spam, non_spam_sample]).sample(frac=1, random_state=42)

# Separazione delle caratteristiche e delle etichette
X = data_bal.iloc[:, :54]  # Prime 54 colonne
y = data_bal["label"]  # Ultima colonna

# Trasformazione TF-IDF
tfidf = TfidfTransformer()
X_tfidf = tfidf.fit_transform(X).toarray()

# Modelli e parametri per Grid Search
models_with_params = {
    "Linear SVM": (SVC(kernel="linear", random_state=42), {"C": [0.1, 1, 10]}),
    "Polynomial SVM (deg 2)": (SVC(kernel="poly", degree=2, random_state=42), {"C": [0.1, 1, 10]}),
    "RBF SVM": (SVC(kernel="rbf", random_state=42), {"C": [0.1, 1, 10], "gamma": [0.1, 1, 10]}),
    "Random Forest": (RandomForestClassifier(random_state=42), {"n_estimators": [50, 100]}),
    "k-NN": (KNeighborsClassifier(n_neighbors=5), {"p": [1, 2]}),
    "GaussianNB": (GaussianNB(), {}),  # Nessuna Grid Search
}



# Liste per salvare i risultati
results = []
model_names = []
trained_models = []
std_accuracies = []
train_times = []
test_times = []

# Lista dei punteggi da calcolare
scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score)
}

# Esecuzione della Grid Search e valutazione dei modelli
for model_name, (model, param_grid) in models_with_params.items():
   print(f"\nEvaluating: {model_name}")
    
   
   # Grid Search con cross-validation
   grid_search = GridSearchCV(model, param_grid, cv=10, scoring=scoring, n_jobs=-1, refit='accuracy')
   grid_search.fit(X_tfidf, y)
        
   best_model = grid_search.best_estimator_ #modello riaddetstrato con i best parametri
   
   best_params = grid_search.best_params_
   
   best_score = grid_search.best_score_ #accuracy media del modello con i parametri migliori (scelti in base all'accuracy)
   std_accuracy = grid_search.cv_results_['std_test_accuracy'][grid_search.best_index_] #deviazione standard dell'accuracy
  
   # Per accedere alle altre metriche per i migliori parametri
   best_precision = np.mean(grid_search.cv_results_['mean_test_precision'][grid_search.best_index_])
   best_recall = np.mean(grid_search.cv_results_['mean_test_recall'][grid_search.best_index_])
   best_f1 = np.mean(grid_search.cv_results_['mean_test_f1'][grid_search.best_index_])
   # Tempi di training e test per il modello con i parametri migliori
   best_train_time = grid_search.cv_results_['mean_fit_time'][grid_search.best_index_]
   best_test_time = grid_search.cv_results_['mean_score_time'][grid_search.best_index_]

   print(f" Best Parameters: {best_params}")
   print(" Scores from Cross Validation of the model with the best parameters combination")
   print(f" mean Accuracy: {best_score:.4f}")
   print(f" mean Precision: {best_precision:.4f}")
   print(f" mean Recall: {best_recall:.4f}")
   print(f" mean F1-Score: {best_f1:.4f}")
   print(f" mean Training Time: {best_train_time:.4f} seconds")
   print(f" mean Test Time: {best_test_time:.4f} seconds")
    
   # Salvare risultati
   results.append(best_score)
   model_names.append(model_name)
   trained_models.append(best_model)
   std_accuracies.append(std_accuracy)
   train_times.append(best_train_time)
   test_times.append(best_test_time)

# Plot dell'Accuracy con deviazione standard
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(model_names, results, color="skyblue", xerr=std_accuracies, capsize=5)
ax.set_xlabel("Accuracy")
ax.set_title("Model Comparison - mean Accuracy obtained from CV")
for i, v in enumerate(results):
    ax.text(v + 0.01, i, f"{v:.3f}", va="center")
plt.tight_layout()
plt.show()

# Plot dei Tempi di Training e Test
fig, ax = plt.subplots(figsize=(10, 6))

# Posizionamento delle barre
bar_width = 0.35
index = np.arange(len(model_names))

# Barre per il training e il test
bars_train = ax.barh(index, train_times, bar_width, label="Training Time", color="#FF6666")
bars_test = ax.barh(index + bar_width, test_times, bar_width, label="Test Time", color="lightgreen")

# Etichette
ax.set_xlabel("Time (seconds)")
ax.set_title("Model Comparison - mean Training and Test Time obtained from CV 10 folds")
ax.set_yticks(index + bar_width / 2)
ax.set_yticklabels(model_names)
ax.legend()

# Etichette dei valori
for bars in [bars_train, bars_test]:
    for bar in bars:
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.3f}", va="center")

plt.tight_layout()
plt.show()

# 📈 DL\_Engineering\_TP2 : Amélioration des Réseaux de Neurones Profonds

## 🌟 Aperçu du Projet

Ce dépôt contient l'implémentation du Travail Pratique 2 (*TP 2*), axé sur l'application des **pratiques d'ingénierie avancées** pour optimiser, régulariser et stabiliser les modèles de Deep Learning.

Le TP vise à diagnostiquer les problèmes de performance (Bias/Variance) et à implémenter des solutions modernes.

### Objectifs Clés

  * **Régularisation :** Maîtrise de **L2** et **Dropout**.
  * **Normalisation :** Utilisation de **Batch Normalization (BN)** pour stabiliser l'entraînement.
  * **Optimisation :** Comparaison des algorithmes (**Adam**, RMSprop, SGD) avec suivi **MLflow**.

### Jeu de Données

  * **MNIST** (classification de 10 catégories d'images de chiffres manuscrits).

## 🛠️ Prise en Main et Structure

### 1\. Structure du Dépôt

```
DL_Engineering_TP2/
├── mlruns/                  # Dossier de suivi MLflow généré
├── train_improved_model.py  # Script d'entraînement principal (avec toutes les modifications)
├── Deep_Learning_Engineering_Report_TP2.pdf  
├── requirements.txt         # Dépendances Python
└── README.md                # Ce fichier
```

### 2\. Configuration et Lancement

1.  **Configuration :** Installez les dépendances :

    ```bash
    pip install -r requirements.txt
    ```

2.  **Entraînement :** Exécutez le script. Il lancera l'entraînement du modèle final et exécutera la boucle de comparaison des optimiseurs, enregistrant tout dans MLflow :

    ```bash
    python train_improved_model.py
    ```

3.  **Visualisation :** Lancez l'interface MLflow pour comparer les courbes de perte (Loss) et de précision (Accuracy) des différents optimiseurs (Ex. 2.3) :

    ```bash
    python -m mlflow ui
    ```

    Accédez à l'interface sur **`http://127.0.0.1:5000`**.

-----

## 📚 Rapport de TP

Le rapport final (fichier LaTeX/Overleaf) répond en détail à toutes les questions théoriques du TP 2 et inclut l'analyse comparative des expérimentations loggées dans MLflow.

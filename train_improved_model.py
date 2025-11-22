import tensorflow as tf
from tensorflow import keras
import numpy as np
import mlflow
import mlflow.tensorflow

# --- Définition des Paramètres ---
EPOCHS = 10 # Augmentation pour l'analyse des optimiseurs
BATCH_SIZE = 128
DROPOUT_RATE = 0.2
MLFLOW_EXPERIMENT_NAME = "MNIST_MLP_Optimization_and_BN"

# --- 1. Chargement et Préparation des Données ---
print("--- 1. Chargement du jeu de données MNIST ---")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalisation et Flattening
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

def create_model(optimizer, dropout_rate, use_bn=False):
    """Crée et compile le modèle MLP."""
    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    ])
    
    if use_bn:
        # Exercice 4: Ajout de la Batch Normalization
        model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- 2. Lancement du Suivi MLflow ---
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# --- 3. Exercice 2: Comparaison des Optimiseurs ---
optimizers_to_test = {
    "Adam": tf.keras.optimizers.Adam(learning_rate=0.001),
    "SGD_Momentum": tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    "RMSprop": tf.keras.optimizers.RMSprop(learning_rate=0.001)
}

print("--- 2. Lancement des Expérimentations ---")

for opt_name, optimizer in optimizers_to_test.items():
    run_name = f"Optimizer_Test_{opt_name}"
    
    # Utilisez un Run imbriqué pour garder l'organisation
    with mlflow.start_run(run_name=run_name, nested=True):
        print(f"\n[Démarrage] Entraînement avec l'optimiseur: {opt_name}")
        
        # Enregistrement des paramètres
        mlflow.log_param("optimizer_name", opt_name)
        mlflow.log_param("epochs", EPOCHS)
        
        # Création et Entraînement du Modèle
        model = create_model(optimizer, DROPOUT_RATE, use_bn=False) # Test sans BN d'abord
        
        # Enregistrement automatique des métriques d'époque
        mlflow.tensorflow.autolog(log_models=False)
        
        history = model.fit(
            x_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.1,
            verbose=0 # Entraînement silencieux pour les comparaisons rapides
        )
        
        # Évaluation et Log final
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        mlflow.log_metric("final_test_accuracy", test_acc)
        mlflow.log_metric("final_test_loss", test_loss)

        print(f"[Terminé] {opt_name} -> Précision Test: {test_acc:.4f}")

# --- 4. Exercice 4: Batch Normalization (BN) ---
print("\n--- 3. Exercice 4: Test avec Batch Normalization (Adam) ---")

with mlflow.start_run(run_name="Adam_with_BatchNormalization"):
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    mlflow.log_param("optimizer_name", "Adam")
    mlflow.log_param("use_batch_normalization", True)
    
    # Création du Modèle AVEC Batch Normalization
    model_bn = create_model(optimizer, DROPOUT_RATE, use_bn=True)
    
    mlflow.tensorflow.autolog(log_models=False)
    
    # Nous utilisons le même nombre d'époques pour la comparaison
    history_bn = model_bn.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1 # Affichage détaillé pour l'analyse
    )
    
    test_loss_bn, test_acc_bn = model_bn.evaluate(x_test, y_test, verbose=0)
    mlflow.log_metric("final_test_accuracy", test_acc_bn)
    mlflow.log_metric("final_test_loss", test_loss_bn)
    
    print(f"\n[BN Terminé] Adam + BN -> Précision Test: {test_acc_bn:.4f}")
    
    # Sauvegarde du meilleur modèle (BN) pour référence
    model_path_local = "best_model_bn.h5" 
    model_bn.save(model_path_local)
    
    mlflow.keras.log_model(
        model=model_bn,
        artifact_path="best_model_bn",
        registered_model_name="MNIST_MLP_Best_Model_BN"
    )
    
print("\nToutes les expérimentations sont enregistrées dans MLflow. Lancez 'mlflow ui' pour l'analyse.")
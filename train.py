
"""Entrenamiento y comparación de dos redes neuronales densas."""

import json
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split

from data import SEED, load_handwritten_normalized


MODEL_PATH = Path("models/digits_model.keras")
RESULTS_PATH = Path("models")


# 1. Semillas para reproducibilidad
def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# 2. Primera red neuronal: API Sequential
def build_sequential():
    model = keras.Sequential([
        keras.layers.Input(shape=(64,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ], name="modelo_sequential")

    return model


# 3. Segunda red neuronal: API funcional
def build_functional():

    entradas = keras.Input(shape=(64,), name="entrada")

    x = keras.layers.Dense(128, activation="relu")(entradas)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation="relu")(x)

    salidas = keras.layers.Dense(
        10,
        activation="softmax",
        name="clasificacion"
    )(x)

    model = keras.Model(
        inputs=entradas,
        outputs=salidas,
        name="modelo_funcional"
    )

    return model


# 4. Entrenamiento de cada modelo
def entrenar(model, nombre, Xtr, ytr, Xva, yva):

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        Xtr,
        ytr,
        validation_data=(Xva, yva),
        epochs=100,
        batch_size=128,
        callbacks=[early_stop],
        verbose=0
    )

    # Evaluación con validación
    pred = model.predict(Xva, verbose=0).argmax(axis=1)

    accuracy = accuracy_score(yva, pred)
    f1 = f1_score(yva, pred, average="macro")

    return {
        "model": model,
        "history": history,
        "name": nombre,
        "val_accuracy": float(accuracy),
        "val_f1_macro": float(f1),
        "parameters": int(model.count_params()),
        "epochs": len(history.history["loss"])
    }


# 5. Programa principal
def main():

    set_seeds()

    RESULTS_PATH.mkdir(exist_ok=True)

    (X, y), (Xte, yte) = load_handwritten_normalized()
    Xtr, Xva, ytr, yva = train_test_split(
        X,
        y,
        test_size=0.10,
        stratify=y,
        random_state=SEED
    )

    print("\n========== DATOS ==========")
    print("Entrenamiento MNIST:", len(ytr))
    print("Validación MNIST:", len(yva))
    print("Prueba MNIST:", len(yte))

    # Crear y entrenar los dos modelos
    resultados = []

    modelos = [
        ("Sequential", build_sequential),
        ("API Funcional", build_functional)
    ]

    for nombre, constructor in modelos:

        print(f"\nEntrenando modelo: {nombre}")

        resultado = entrenar(
            constructor(),
            nombre,
            Xtr, ytr,
            Xva, yva
        )

        resultados.append(resultado)

        print("Accuracy validación:",
              f"{resultado['val_accuracy']:.2%}")

        print("F1 macro:",
              f"{resultado['val_f1_macro']:.4f}")

        print("Parámetros:",
              resultado["parameters"])

        print("Épocas efectivas:",
              resultado["epochs"])

    # 6. Seleccionar modelo usando únicamente validación
    ganador = max(
        resultados,
        key=lambda r: r["val_f1_macro"]
    )

    print("\n========== MODELO SELECCIONADO ==========")
    print("Modelo:", ganador["name"])

    # 7. Evaluación final de ambos modelos en prueba
    metricas = {}

    for resultado in resultados:

        pred = resultado["model"].predict(
            Xte,
            verbose=0
        ).argmax(axis=1)

        test_accuracy = accuracy_score(yte, pred)
        test_f1 = f1_score(yte, pred, average="macro")

        metricas[resultado["name"]] = {
            "val_accuracy": resultado["val_accuracy"],
            "val_f1_macro": resultado["val_f1_macro"],
            "test_accuracy": float(test_accuracy),
            "test_f1_macro": float(test_f1),
            "parameters": resultado["parameters"],
            "epochs": resultado["epochs"]
        }

        print(f"\n{resultado['name']}")
        print("Accuracy prueba:", f"{test_accuracy:.2%}")
        print("F1 macro prueba:", f"{test_f1:.4f}")

    # 8. Guardar el modelo ganador
    ganador["model"].save(MODEL_PATH)

    configuracion = {
        "seed": int(SEED),
        "selected_model": ganador["name"],
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "batch_size": 128,
        "dataset": "MNIST reducido a 8x8",
        "max_epochs": 100,
        "early_stopping_patience": 10,
        "models": metricas
    }

    with open(
        RESULTS_PATH / "metrics.json",
        "w",
        encoding="utf-8"
    ) as archivo:
        json.dump(configuracion, archivo, indent=4)

    # 9. Curvas de pérdida y matrices de confusión
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for i, resultado in enumerate(resultados):

        hist = resultado["history"].history

        axes[i, 0].plot(
            hist["loss"],
            label="Entrenamiento"
        )

        axes[i, 0].plot(
            hist["val_loss"],
            label="Validación"
        )

        axes[i, 0].set_title(
            f"Pérdida - {resultado['name']}"
        )

        axes[i, 0].set_xlabel("Épocas")
        axes[i, 0].set_ylabel("Pérdida")
        axes[i, 0].legend()

        pred = resultado["model"].predict(
            Xte,
            verbose=0
        ).argmax(axis=1)

        ConfusionMatrixDisplay(
            confusion_matrix(yte, pred),
            display_labels=range(10)
        ).plot(
            ax=axes[i, 1],
            colorbar=False
        )

        axes[i, 1].set_title(
            f"Matriz de confusión - {resultado['name']}"
        )

    plt.tight_layout()

    plt.savefig(
        RESULTS_PATH / "resultados_entrenamiento.png",
        dpi=150
    )

    plt.close()

    print("\n====================================")
    print("ENTRENAMIENTO FINALIZADO")
    print("Modelo guardado:", MODEL_PATH)
    print("Métricas guardadas: models/metrics.json")
    print("Gráficos guardados: models/resultados_entrenamiento.png")
    print("====================================")


if __name__ == "__main__":
    main()

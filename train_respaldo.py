"""Entrena la red densa, decide con validación y evalúa en prueba UNA sola vez."""
import json
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from data import SEED, get_splits, save_indices

MODEL_PATH = Path("models/digits_model.keras")


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_model():
    return keras.Sequential([
        keras.layers.Input(shape=(64,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ], name="digitos_denso")


def main():
    set_seeds()
    (Xtr, ytr), (Xva, yva), (Xte, yte), (itr, iva, ite) = get_splits()
    save_indices(itr, iva, ite)
    print(f"Tamaños -> train {len(ytr)}, val {len(yva)}, test {len(yte)}")

    model = build_model()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True)
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=200,
                     batch_size=32, callbacks=[stop], verbose=0)

    _, val_acc = model.evaluate(Xva, yva, verbose=0)
    _, test_acc = model.evaluate(Xte, yte, verbose=0)  # única evaluación en prueba
    print(f"Exactitud validación: {val_acc:.2%} | prueba: {test_acc:.2%}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save(MODEL_PATH)
    Path("models/metrics.json").write_text(json.dumps(
        {"val_accuracy": val_acc, "test_accuracy": test_acc,
         "epochs_run": len(hist.history["loss"]), "seed": SEED}, indent=2))

    # Curvas y matriz de confusión
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist.history["loss"], label="train")
    ax[0].plot(hist.history["val_loss"], label="val")
    ax[0].set(title="Pérdida", xlabel="Época"); ax[0].legend()
    pred = model.predict(Xte, verbose=0).argmax(axis=1)
    ConfusionMatrixDisplay(confusion_matrix(yte, pred)).plot(ax=ax[1], colorbar=False)
    ax[1].set_title("Matriz de confusión (prueba)")
    plt.tight_layout(); plt.savefig("models/resultados_entrenamiento.png", dpi=150)


if __name__ == "__main__":
    main()

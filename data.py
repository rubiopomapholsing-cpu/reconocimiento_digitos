"""Preparación reproducible de datos: load_digits, normalización 0-1 y división 60/20/20."""
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 42
MAX_PIXEL = 16.0  # escala original de load_digits: 0..16
SPLITS_PATH = Path("models/splits.json")


def load_normalized():
    """Devuelve X en [0, 1] (n, 64) e y (n,)."""
    digits = load_digits()
    return digits.data / MAX_PIXEL, digits.target


def load_handwritten_normalized(batch_size=2048):
    """Carga MNIST y reduce sus imágenes a 8x8 para la app de escritura libre."""
    import tensorflow as tf
    from tensorflow import keras

    (images_train, y_train), (images_test, y_test) = keras.datasets.mnist.load_data()

    def resize(images):
        batches = []
        for start in range(0, len(images), batch_size):
            image_batch = images[start:start + batch_size, ..., np.newaxis]
            small = tf.image.resize(image_batch, (8, 8), method="area")
            batches.append(small.numpy().reshape(-1, 64) / 255.0)
        return np.concatenate(batches).astype(np.float32)

    return (resize(images_train), y_train), (resize(images_test), y_test)


def split_indices(y, seed=SEED):
    """Índices estratificados 60/20/20 (entrenamiento/validación/prueba)."""
    idx = np.arange(len(y))
    idx_train, idx_tmp = train_test_split(
        idx, test_size=0.40, stratify=y, random_state=seed)
    idx_val, idx_test = train_test_split(
        idx_tmp, test_size=0.50, stratify=y[idx_tmp], random_state=seed)
    return idx_train, idx_val, idx_test


def get_splits(seed=SEED):
    X, y = load_normalized()
    tr, va, te = split_indices(y, seed)
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te]), (tr, va, te)


def save_indices(tr, va, te, path=SPLITS_PATH):
    """Conserva los índices para poder auditar la separación."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"seed": SEED, "train": tr.tolist(), "val": va.tolist(), "test": te.tolist()}))

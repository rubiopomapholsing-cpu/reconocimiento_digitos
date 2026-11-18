import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytest
from data import load_normalized, split_indices, load_handwritten_normalized
from preprocess import canvas_to_input
from storage import log_result, read_results


def test_normalizacion_0_1():
    X, y = load_normalized()
    assert X.shape == (1797, 64) and X.min() == 0 and X.max() == 1.0


def test_proporciones_60_20_20():
    _, y = load_normalized()
    tr, va, te = split_indices(y)
    n = len(y)
    assert abs(len(tr) / n - 0.6) < 0.01
    assert abs(len(va) / n - 0.2) < 0.01
    assert abs(len(te) / n - 0.2) < 0.01


def test_sin_solapamiento_y_cobertura():
    _, y = load_normalized()
    tr, va, te = split_indices(y)
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    assert len(set(tr) | set(va) | set(te)) == len(y)


def test_estratificacion():
    _, y = load_normalized()
    tr, va, te = split_indices(y)
    glob = np.bincount(y) / len(y)
    for idx in (tr, va, te):
        assert np.allclose(np.bincount(y[idx], minlength=10) / len(idx), glob, atol=0.02)


def test_reproducible():
    _, y = load_normalized()
    a, b = split_indices(y), split_indices(y)
    assert all(np.array_equal(x, z) for x, z in zip(a, b))


def test_preprocesamiento_lienzo():
    c = np.zeros((280, 280)); c[:, 130:150] = 1.0
    x = canvas_to_input(c)
    assert x.shape == (1, 64) and 0 <= x.min() and x.max() == 1.0
    assert canvas_to_input(np.zeros((280, 280))).sum() == 0


def test_invariante_a_posicion_y_tamano():
    c1 = np.zeros((280, 280)); c1[40:200, 100:130] = 1     # un "1" grande
    c2 = np.zeros((280, 280)); c2[150:230, 200:215] = 1    # mismo trazo, chico y desplazado
    assert np.abs(canvas_to_input(c1) - canvas_to_input(c2)).max() < 0.35


def test_persistencia_csv(tmp_path):
    p = tmp_path / "log.csv"
    log_result(7, 0.93, "7", path=p); log_result(1, 0.8, path=p)
    rows = read_results(p)
    assert len(rows) == 2 and rows[0]["prediccion"] == "7"


def test_modelo_supera_meta():
    pytest.importorskip("tensorflow")
    from train import MODEL_PATH
    if not MODEL_PATH.exists():
        pytest.skip("Entrena primero con train.py")
    from tensorflow import keras
    (_, _), (Xte, yte) = load_handwritten_normalized()
    acc = (keras.models.load_model(MODEL_PATH).predict(Xte, verbose=0).argmax(1) == yte).mean()
    assert acc >= 0.90

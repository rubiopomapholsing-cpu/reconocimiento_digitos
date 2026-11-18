
"""Persistencia: registro de resultados en CSV."""

import csv
from datetime import datetime
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent / "resultados.csv"

FIELDS = [
    "fecha",
    "prediccion",
    "confianza",
    "valor_real",
    "correcto",
    "incorrecto"
]


def log_result(prediction, confidence, true_value="", path=LOG_PATH):
    true_value = str(true_value).strip()

    if true_value and (len(true_value) != 1 or not "0" <= true_value <= "9"):
        raise ValueError("El dígito real debe estar entre 0 y 9.")

    prediction = int(prediction)
    correcto = prediction == int(true_value) if true_value else ""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not file_path.exists() or file_path.stat().st_size == 0

    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if needs_header:
            writer.writeheader()

        writer.writerow({
            "fecha": datetime.now().isoformat(timespec="seconds"),
            "prediccion": prediction,
            "confianza": round(float(confidence), 4),
            "valor_real": true_value,
            "correcto": correcto,
            "incorrecto": not correcto if correcto != "" else ""
        })


def read_results(path=LOG_PATH):
    if not Path(path).exists():
        return []

    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

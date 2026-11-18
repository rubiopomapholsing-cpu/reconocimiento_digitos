
import numpy as np
import tensorflow as tf

from sklearn.metrics import confusion_matrix
from data import load_handwritten_normalized

# Cargar los datos
(_, _), (Xte, yte) = load_handwritten_normalized()

# Cargar el modelo seleccionado
modelo = tf.keras.models.load_model(
    "models/digits_model.keras"
)

# Obtener predicciones
probabilidades = modelo.predict(Xte, verbose=0)
predicciones = np.argmax(probabilidades, axis=1)

# Buscar errores
errores = np.where(predicciones != yte)[0]

print("\n========== ANÁLISIS DE ERRORES ==========")
print("Total de imágenes incorrectas:", len(errores))

if len(errores) == 0:
    print("No se encontraron errores.")
else:
    for i, indice in enumerate(errores[:3], start=1):

        real = int(yte[indice])
        predicho = int(predicciones[indice])
        confianza = float(
            probabilidades[indice][predicho]
        )

        print(f"\nError {i}")
        print("Dígito real:", real)
        print("Dígito predicho:", predicho)
        print("Confianza:", f"{confianza:.2%}")

        print("Imagen 8x8:")
        print(
            np.array2string(
                Xte[indice].reshape(8, 8),
                precision=2
            )
        )

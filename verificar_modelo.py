
import numpy as np
import tensorflow as tf

from data import load_handwritten_normalized

# 1. Cargar nuevamente los datos de prueba
(_, _), (X_test, y_test) = load_handwritten_normalized()

# 2. Cargar el modelo guardado
modelo = tf.keras.models.load_model(
    "models/digits_model.keras"
)

# 3. Seleccionar cinco imágenes fijas
indices = [0, 1, 2, 3, 4]

# 4. Realizar predicciones
print("\n===== VERIFICACIÓN DEL MODELO RECARGADO =====")

for indice in indices:
    imagen = X_test[indice].reshape(1, 64)

    probabilidades = modelo.predict(
        imagen,
        verbose=0
    )

    prediccion = int(np.argmax(probabilidades))
    real = int(y_test[indice])
    confianza = float(np.max(probabilidades))

    print(f"\nImagen de prueba: {indice}")
    print("Dígito real:", real)
    print("Dígito predicho:", prediccion)
    print("Confianza:", f"{confianza:.2%}")

print("\nVerificación finalizada correctamente.")

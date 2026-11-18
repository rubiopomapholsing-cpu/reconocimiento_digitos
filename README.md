# Sistema interactivo de reconocimiento de dígitos

Aplicación local con una interfaz Tkinter y una red densa de Keras. El modelo
se entrena con MNIST, un conjunto de imágenes de escritura manual, reducidas a
8×8 píxeles para conservar el formato compacto que usa la interfaz.

## Estructura

- `data.py`: carga MNIST y reduce las imágenes a vectores 8×8; conserva también
  `load_digits` para los análisis del conjunto de datos original.
- `train.py`: compara dos redes, evalúa en MNIST y guarda el mejor modelo y sus
  métricas en `models/`.
- `app.py`: permite dibujar, reconocer, consultar probabilidades y guardar
  resultados.
- `preprocess.py`: centra el trazo y lo convierte al formato 8×8 del modelo.
- `storage.py`: agrega los registros a `resultados.csv`, incluso si aún no se
  conoce el dígito real.

## Uso

```text
pip install -r requirements.txt
python train.py     # la primera vez descarga MNIST y entrena el modelo
python app.py       # carga el modelo guardado
```

El modelo actual obtuvo 97,42 % de exactitud en el conjunto de prueba de MNIST
después de reducir las imágenes a 8×8.

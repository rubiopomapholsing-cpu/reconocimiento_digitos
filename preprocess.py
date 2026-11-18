"""Convierte un dibujo al formato 8x8 normalizado usado durante el entrenamiento.

El trazo se recorta, se ajusta sin deformarlo a una zona de 6x6 y se centra
en el lienzo 8x8 para dejar un margen similar al de las imágenes de entrenamiento.
"""
from collections import deque

import numpy as np
from scipy import ndimage


def canvas_to_input(canvas: np.ndarray) -> np.ndarray:
    """canvas: matriz 2D con valores 0..1 (1 = trazo), de cualquier tamaño.
    Devuelve un vector (1, 64) en [0, 1]."""
    ys, xs = np.nonzero(canvas > 0.05)
    if len(ys) == 0:
        return np.zeros((1, 64), dtype=np.float32)

    crop = canvas[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
    h, w = crop.shape
    side = max(h, w)
    square = np.zeros((side, side), dtype=np.float32)
    top, left = (side - h) // 2, (side - w) // 2
    square[top:top + h, left:left + w] = crop

    # Reduce la imagen por promedio de áreas antes de colocarla en el 8x8.
    target = 6
    up = np.repeat(np.repeat(square, target, axis=0), target, axis=1)
    blocks = up.reshape(target, side, target, side).mean(axis=(1, 3))

    # Deja un borde de un píxel alrededor del dígito.
    image = np.zeros((8, 8), dtype=np.float32)
    image[1:7, 1:7] = blocks
    max_value = float(image.max())
    if max_value > 0:
        image /= max_value
    return np.clip(image, 0, 1).reshape(1, 64).astype(np.float32)


def looks_like_nine(canvas: np.ndarray) -> bool:
    """Detecta el patrón de un 9: un bucle cerrado arriba y una cola larga abajo."""
    ink = np.asarray(canvas) > 0.5
    ink = ndimage.binary_closing(ink, structure=np.ones((7, 7), dtype=bool))
    ys, xs = np.nonzero(ink)
    if len(ys) < 20:
        return False

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    digit = ink[y0:y1, x0:x1]
    height, width = digit.shape
    if height < 1.1 * width:
        return False

    background = ~digit
    outside = np.zeros_like(background)
    queue = deque()

    def add_background(y, x):
        if background[y, x] and not outside[y, x]:
            outside[y, x] = True
            queue.append((y, x))

    for x in range(width):
        add_background(0, x)
        add_background(height - 1, x)
    for y in range(height):
        add_background(y, 0)
        add_background(y, width - 1)

    while queue:
        y, x = queue.popleft()
        if y > 0:
            add_background(y - 1, x)
        if y + 1 < height:
            add_background(y + 1, x)
        if x > 0:
            add_background(y, x - 1)
        if x + 1 < width:
            add_background(y, x + 1)

    hole_y, hole_x = np.nonzero(background & ~outside)
    if len(hole_y) < max(20, int(height * width * 0.01)):
        return False

    hole_top = int(hole_y.min())
    hole_bottom = int(hole_y.max())
    hole_center_y = float(hole_y.mean())
    if hole_center_y > height * 0.55 or hole_bottom >= height - 1:
        return False

    lower_start = max(hole_bottom + 1, int(height * 0.55))
    lower_rows = np.flatnonzero(np.any(digit[lower_start:, :], axis=1))
    if len(lower_rows) == 0:
        return False

    lower_rows += lower_start
    row_widths = []
    row_centers = []
    for row in lower_rows:
        columns = np.flatnonzero(digit[row])
        row_widths.append(len(columns))
        row_centers.append(float(columns.mean()))

    return (
        hole_top < height * 0.45
        and int(lower_rows.max()) - int(lower_rows.min()) >= height * 0.20
        and float(np.median(row_centers)) >= width * 0.50
        and float(np.median(row_widths)) <= width * 0.45
    )

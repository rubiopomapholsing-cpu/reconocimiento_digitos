
"""Reconocimiento Inteligente de Dígitos - Interfaz moderna Tkinter."""

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

from preprocess import canvas_to_input, looks_like_nine
from storage import log_result, read_results
from train import MODEL_PATH


SIZE = 280
BRUSH = 9
UMBRAL_CONFIANZA = 0.70


class DigitApp:

    def __init__(self, root, model):
        self.root = root
        self.model = model

        self.grid = np.zeros((SIZE, SIZE), dtype=np.float32)
        self.last_pred = None
        self.last_input = None
        self.last_probs = None

        self.root.title("Reconocimiento Inteligente de Dígitos")
        self.root.geometry("1150x850")
        self.root.minsize(1050, 750)
        self.root.configure(bg="#eef2f7")

        self.setup_styles()
        self.build_interface()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(
            "TButton",
            font=("Segoe UI", 10, "bold"),
            padding=9
        )

        style.configure(
            "Primary.TButton",
            background="#2563eb",
            foreground="white"
        )

        style.map(
            "Primary.TButton",
            background=[("active", "#1d4ed8")]
        )

        style.configure(
            "TEntry",
            padding=7,
            font=("Segoe UI", 11)
        )

    def build_interface(self):

        # ENCABEZADO
        header = tk.Frame(self.root, bg="#172554", height=85)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="RECONOCIMIENTO INTELIGENTE DE DÍGITOS",
            bg="#172554",
            fg="white",
            font=("Segoe UI", 19, "bold")
        ).pack(anchor="w", padx=25, pady=(13, 2))

        tk.Label(
            header,
            text="Sistema de clasificación mediante Deep Learning",
            bg="#172554",
            fg="#bfdbfe",
            font=("Segoe UI", 10)
        ).pack(anchor="w", padx=27)

        # CONTENEDOR PRINCIPAL
        main = tk.Frame(self.root, bg="#eef2f7")
        main.pack(fill="both", expand=True, padx=20, pady=15)

        left = tk.Frame(main, bg="white", padx=15, pady=15)
        left.pack(side="left", fill="y", padx=(0, 12))

        right = tk.Frame(main, bg="#eef2f7")
        right.pack(side="left", fill="both", expand=True)

        # PANEL DE DIBUJO
        tk.Label(
            left,
            text="Dibuja un dígito",
            bg="white",
            fg="#172554",
            font=("Segoe UI", 16, "bold")
        ).pack(anchor="w", pady=(0, 5))

        tk.Label(
            left,
            text="Utiliza el mouse para dibujar del 0 al 9",
            bg="white",
            fg="#64748b",
            font=("Segoe UI", 9)
        ).pack(anchor="w", pady=(0, 12))

        self.canvas = tk.Canvas(
            left,
            width=SIZE,
            height=SIZE,
            bg="black",
            cursor="cross",
            highlightthickness=2,
            highlightbackground="#cbd5e1"
        )
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self.draw)

        buttons = tk.Frame(left, bg="white")
        buttons.pack(fill="x", pady=15)

        ttk.Button(
            buttons,
            text="Limpiar",
            command=self.clear
        ).pack(side="left", fill="x", expand=True, padx=(0, 5))

        ttk.Button(
            buttons,
            text="Reconocer",
            style="Primary.TButton",
            command=self.predict
        ).pack(side="left", fill="x", expand=True, padx=(5, 0))

        tk.Label(
            left,
            text="● MODELO CARGADO",
            bg="white",
            fg="#15803d",
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", pady=(5, 0))

        # RESULTADO
        result_card = tk.Frame(right, bg="white", padx=18, pady=12)
        result_card.pack(fill="x", pady=(0, 10))

        tk.Label(
            result_card,
            text="Resultado del reconocimiento",
            bg="white",
            fg="#64748b",
            font=("Segoe UI", 11)
        ).pack(anchor="w")

        self.result = tk.Label(
            result_card,
            text="--",
            bg="white",
            fg="#2563eb",
            font=("Segoe UI", 34, "bold")
        )
        self.result.pack(anchor="w")

        self.confidence_label = tk.Label(
            result_card,
            text="Confianza: --",
            bg="white",
            fg="#334155",
            font=("Segoe UI", 11, "bold")
        )
        self.confidence_label.pack(anchor="w")

        self.warning = tk.Label(
            result_card,
            text="Dibuja un número y pulsa Reconocer.",
            bg="white",
            fg="#64748b",
            font=("Segoe UI", 10),
            wraplength=650,
            justify="left"
        )
        self.warning.pack(anchor="w", pady=(5, 0))

        # PANEL DE INFORMACIÓN
        details = tk.Frame(right, bg="#eef2f7")
        details.pack(fill="both", expand=True)

        # TOP 3
        predictions_card = tk.Frame(
            details,
            bg="white",
            padx=15,
            pady=10
        )
        predictions_card.pack(fill="x", pady=(0, 8))

        tk.Label(
            predictions_card,
            text="Top 3 predicciones",
            bg="white",
            fg="#172554",
            font=("Segoe UI", 12, "bold")
        ).pack(anchor="w")

        self.top3 = tk.Label(
            predictions_card,
            text="1. --\n2. --\n3. --",
            bg="white",
            fg="#334155",
            font=("Segoe UI", 10),
            justify="left"
        )
        self.top3.pack(anchor="w", pady=5)

        # PANEL DE REGISTRO: AHORA DEBAJO DEL TOP 3
        register = tk.Frame(
            details,
            bg="white",
            padx=15,
            pady=10
        )
        register.pack(fill="x", pady=(0, 8))

        tk.Label(
            register,
            text="Registrar resultado",
            bg="white",
            fg="#172554",
            font=("Segoe UI", 12, "bold")
        ).pack(anchor="w", pady=(0, 7))

        form = tk.Frame(register, bg="white")
        form.pack(fill="x")

        tk.Label(
            form,
            text="Dígito real (0-9, opcional):",
            bg="white",
            fg="#334155",
            font=("Segoe UI", 10)
        ).pack(side="left")

        self.truth = ttk.Entry(form, width=6)
        self.truth.pack(side="left", padx=8)

        ttk.Button(
            form,
            text="Guardar resultado",
            style="Primary.TButton",
            command=self.save
        ).pack(side="left", padx=5)

        ttk.Button(
            form,
            text="Ver historial",
            command=self.history
        ).pack(side="left", padx=5)

        # GRÁFICA
        chart_card = tk.Frame(
            details,
            bg="white",
            padx=15,
            pady=8
        )
        chart_card.pack(fill="both", expand=True, pady=(0, 8))

        tk.Label(
            chart_card,
            text="Probabilidades del 0 al 9",
            bg="white",
            fg="#172554",
            font=("Segoe UI", 12, "bold")
        ).pack(anchor="w")

        self.chart = tk.Canvas(
            chart_card,
            height=150,
            bg="white",
            highlightthickness=0
        )
        self.chart.pack(fill="both", expand=True)

        # BARRA INFERIOR
        self.status = tk.Label(
            self.root,
            text="Sistema listo | Modelo local TensorFlow",
            bg="#dbeafe",
            fg="#1e3a8a",
            anchor="w",
            padx=20,
            pady=7,
            font=("Segoe UI", 9)
        )
        self.status.pack(fill="x", side="bottom")

    # DIBUJAR
    def draw(self, event):
        r = BRUSH

        x = max(0, min(SIZE - 1, event.x))
        y = max(0, min(SIZE - 1, event.y))

        self.canvas.create_oval(
            x-r, y-r, x+r, y+r,
            fill="white",
            outline="white"
        )

        yy, xx = np.ogrid[:SIZE, :SIZE]
        mask = (yy-y)**2 + (xx-x)**2 <= r*r
        self.grid[mask] = 1.0

        self.last_pred = None
        self.last_input = None
        self.last_probs = None

        self.result.config(text="--")
        self.confidence_label.config(text="Confianza: --")
        self.warning.config(
            text="Dibujo modificado. Pulsa Reconocer.",
            fg="#64748b"
        )

    # RECONOCER
    def predict(self):

        if self.grid.max() == 0:
            messagebox.showwarning(
                "Lienzo vacío",
                "Primero debes dibujar un dígito."
            )
            return

        entrada = canvas_to_input(self.grid)

        if not np.any(entrada):
            messagebox.showwarning(
                "Dibujo vacío",
                "No se detectaron trazos suficientes."
            )
            return

        probs = self.model.predict(entrada, verbose=0)[0]

        prediction = int(np.argmax(probs))
        confidence = float(probs[prediction])
        corrected_to_nine = (
            prediction == 3
            and float(probs[9]) >= 0.20
            and looks_like_nine(self.grid)
        )
        if corrected_to_nine:
            prediction = 9
            confidence = float(probs[9])

        self.last_pred = (prediction, confidence)
        self.last_input = entrada.copy()
        self.last_probs = probs.copy()

        self.result.config(text=str(prediction))

        self.confidence_label.config(
            text=f"Confianza: {confidence:.2%}"
        )

        if corrected_to_nine:
            self.warning.config(
                text=(
                    "El bucle cerrado y la cola descendente indican un 9. "
                    f"La red confundió el trazo con un 3 ({probs[3]:.2%}); "
                    f"su probabilidad original para 9 es {probs[9]:.2%}."
                ),
                fg="#b45309"
            )
        elif confidence < UMBRAL_CONFIANZA:
            self.warning.config(
                text="Advertencia: confianza inferior al 70%. "
                     "La predicción puede ser poco fiable.",
                fg="#dc2626"
            )
        else:
            self.warning.config(
                text="Predicción con confianza igual o superior al 70%.",
                fg="#15803d"
            )

        # TRES MEJORES PREDICCIONES
        top_indices = np.argsort(probs)[-3:][::-1]

        top_text = "\n".join(
            f"{i+1}. Dígito {int(d)} — {probs[d]:.2%}"
            for i, d in enumerate(top_indices)
        )

        self.top3.config(text=top_text)

        self.draw_chart(probs, prediction)

        self.status.config(
            text=(
                f"Reconocimiento completado | Predicción: {prediction}"
                + (" (ajustada por la forma del trazo)" if corrected_to_nine else "")
            )
        )

    # GRÁFICA DE PROBABILIDADES
    def draw_chart(self, probs, prediction=None):

        self.chart.delete("all")
        self.chart.update_idletasks()

        width = max(self.chart.winfo_width(), 400)
        height = max(self.chart.winfo_height(), 150)

        margin_left = 35
        margin_right = 15
        margin_top = 20
        margin_bottom = 25

        chart_width = width - margin_left - margin_right
        chart_height = height - margin_top - margin_bottom

        bar_width = chart_width / 10

        self.chart.create_line(
            margin_left,
            margin_top + chart_height,
            width - margin_right,
            margin_top + chart_height,
            fill="#94a3b8"
        )

        highlighted_digit = (
            int(np.argmax(probs)) if prediction is None else int(prediction)
        )
        for digit, probability in enumerate(probs):

            x0 = margin_left + digit * bar_width + 5
            x1 = margin_left + (digit + 1) * bar_width - 5

            bar_height = float(probability) * chart_height

            y0 = margin_top + chart_height - bar_height
            y1 = margin_top + chart_height

            color = (
                "#16a34a"
                if digit == highlighted_digit
                else "#2563eb"
            )

            self.chart.create_rectangle(
                x0, y0, x1, y1,
                fill=color,
                outline=""
            )

            self.chart.create_text(
                (x0+x1)/2,
                height - 12,
                text=str(digit),
                fill="#334155",
                font=("Segoe UI", 9)
            )

            if probability > 0.05:
                self.chart.create_text(
                    (x0+x1)/2,
                    max(y0-8, 8),
                    text=f"{probability:.0%}",
                    fill="#334155",
                    font=("Segoe UI", 8)
                )

    # GUARDAR RESULTADO
    def save(self):

        if self.last_pred is None:
            if self.grid.max() == 0:
                messagebox.showwarning(
                    "Sin dibujo",
                    "Primero dibuja un dígito."
                )
                return

            self.predict()
            if self.last_pred is None:
                return

        true_value = self.truth.get().strip()

        if true_value and (len(true_value) != 1 or not "0" <= true_value <= "9"):
            messagebox.showwarning(
                "Valor incorrecto",
                "Ingresa un único dígito entre 0 y 9."
            )
            return

        try:
            log_result(
                self.last_pred[0],
                self.last_pred[1],
                true_value
            )

        except (ValueError, OSError) as error:
            messagebox.showerror(
                "Error al guardar",
                str(error)
            )
            return

        correcto = (
            self.last_pred[0] == int(true_value)
            if true_value
            else None
        )
        resultado = (
            "Correcto" if correcto is True
            else "Incorrecto" if correcto is False
            else "Pendiente: no ingresaste el dígito real"
        )

        messagebox.showinfo(
            "Resultado guardado",
            "El registro se guardó correctamente.\n"
            f"Predicción: {self.last_pred[0]}\n"
            f"Valor real: {true_value or 'no ingresado'}\n"
            f"Resultado: {resultado}"
        )

        self.status.config(
            text="Resultado registrado en resultados.csv"
        )

    # LIMPIAR
    def clear(self):

        self.canvas.delete("all")
        self.grid[:] = 0

        self.last_pred = None
        self.last_input = None
        self.last_probs = None

        self.result.config(text="--")
        self.confidence_label.config(text="Confianza: --")

        self.warning.config(
            text="Dibuja un dígito y pulsa Reconocer.",
            fg="#64748b"
        )

        self.top3.config(text="1. --\n2. --\n3. --")

        self.chart.delete("all")
        self.truth.delete(0, "end")

        self.status.config(text="Lienzo limpiado | Sistema listo")

    # HISTORIAL
    def history(self):

        rows = read_results()

        if not rows:
            messagebox.showinfo(
                "Historial",
                "Todavía no hay registros."
            )
            return

        window = tk.Toplevel(self.root)
        window.title("Historial de reconocimientos")
        window.geometry("850x400")
        window.configure(bg="white")

        tk.Label(
            window,
            text="Historial de reconocimientos",
            bg="white",
            fg="#172554",
            font=("Segoe UI", 16, "bold")
        ).pack(anchor="w", padx=15, pady=12)

        columns = (
            "fecha",
            "prediccion",
            "confianza",
            "valor_real",
            "correcto",
            "incorrecto"
        )

        table = ttk.Treeview(
            window,
            columns=columns,
            show="headings"
        )

        headings = {
            "fecha": "Fecha y hora",
            "prediccion": "Predicción",
            "confianza": "Confianza",
            "valor_real": "Valor real",
            "correcto": "Correcto",
            "incorrecto": "Incorrecto"
        }

        for col in columns:
            table.heading(col, text=headings[col])
            table.column(col, width=125, anchor="center")

        for row in rows[-100:][::-1]:
            table.insert(
                "",
                "end",
                values=tuple(row.get(col, "") for col in columns)
            )

        table.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=(0, 15)
        )


def main():

    from tensorflow import keras

    if not MODEL_PATH.exists():
        raise SystemExit(
            "No se encontró el modelo. Ejecuta primero: python train.py"
        )

    model = keras.models.load_model(MODEL_PATH)

    root = tk.Tk()
    DigitApp(root, model)
    root.mainloop()


if __name__ == "__main__":
    main()

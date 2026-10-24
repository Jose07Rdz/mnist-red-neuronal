# ¿Qué número es? Mi primera red neuronal con MNIST

Práctica para aprender lo básico de redes neuronales con PyTorch: clasificar dígitos escritos a mano (0 a 9).
Incluye una demo web donde dibujas un número y ves cómo lo reconoce la red, capa por capa.

**▶ Demo en vivo: [jose07rdz.github.io/mnist-red-neuronal](https://jose07rdz.github.io/mnist-red-neuronal/)**

## Qué hay aquí

| Archivo | Qué es |
|---|---|
| `mnist.ipynb` | Notebook guiado: datos, red densa (MLP), bucle de entrenamiento, evaluación, red convolucional (CNN) y ejercicios |
| `entrenar_demo.py` | Entrena las dos redes de la demo y exporta sus pesos a `docs/modelos.json` |
| `docs/` | La demo web (GitHub Pages): HTML, CSS y JavaScript puro, sin servidor |

## Resultados

En las 10,000 imágenes de prueba de MNIST:

| Modelo | Arquitectura | Parámetros | Precisión |
|---|---|---|---|
| MLP (notebook) | 784 → 128 → 10 | 102 mil | 97.6 % |
| CNN (notebook) | 2 convoluciones (32 y 64 filtros) → 128 → 10 | 422 mil | 99.0 % |
| MLP (demo) | 784 → 64 → 32 → 10 | 53 mil | 97.0 % |
| CNN (demo) | 2 convoluciones (8 y 16 filtros) → 64 → 10 | 52 mil | 98.7 % |

Los modelos de la demo son más chicos para poder dibujar cada capa, y se entrenan con aumento de datos
(rotación, escala y desplazamiento) porque los números dibujados con el mouse no se parecen del todo a los de MNIST.

## Cómo funciona la demo

1. Tu dibujo se recorta, se reduce a 20×20 píxeles dentro de un cuadro de 28×28 y se centra por su centro de masa,
   el mismo preprocesamiento que tienen las imágenes de MNIST.
2. Las redes se ejecutan en el navegador con JavaScript puro (`docs/red.js`): convolución, ReLU, max pooling y
   capas densas escritas a mano. Con las mismas imágenes dan los mismos resultados que PyTorch (diferencia < 0.000002).
3. El diagrama colorea las activaciones de cada capa y las conexiones que más empujan hacia la siguiente.

## Uso

Requiere [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run jupyter lab mnist.ipynb          # el notebook
uv run python entrenar_demo.py          # reentrenar los modelos de la demo
python3 -m http.server --directory docs # ver la demo en http://localhost:8000
```

La primera ejecución descarga MNIST (~60 MB) a `data/`.

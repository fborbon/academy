# Redes Neuronales 1001: de una neurona a GPT

La edición en español del curso: nueve notebooks de Jupyter que construyen el aprendizaje profundo moderno desde los primeros principios. Cada uno empieza donde terminó el anterior y acaba con algo que entrena y se ejecuta en la CPU de un equipo personal, sin GPU, sin claves de API y sin descargar conjuntos de datos.

**Sitio web del curso:** [neural-networks.academy.forwardforecasting.eu/es/](https://neural-networks.academy.forwardforecasting.eu/es/) muestra cada notebook con sus resultados guardados, el temario, el seguimiento del progreso y enlaces para descargar o abrir en Colab.

| # | Notebook | Tiempo en CPU |
|---|---|---|
| 01 | [`01_la_neurona_artificial.ipynb`](01_la_neurona_artificial.ipynb) | < 30 s |
| 02 | [`02_redes_neuronales_y_retropropagacion.ipynb`](02_redes_neuronales_y_retropropagacion.ipynb) | < 1 min |
| 03 | [`03_entrenar_bien_redes_profundas.ipynb`](03_entrenar_bien_redes_profundas.ipynb) | ~7 min |
| 04 | [`04_redes_convolucionales.ipynb`](04_redes_convolucionales.ipynb) | ~3 min |
| 05 | [`05_redes_recurrentes.ipynb`](05_redes_recurrentes.ipynb) | < 1 min |
| 06 | [`06_embeddings_y_traduccion.ipynb`](06_embeddings_y_traduccion.ipynb) | ~4 min |
| 07 | [`07_transformer.ipynb`](07_transformer.ipynb) | ~3 min |
| 08 | [`08_gpt_desde_cero.ipynb`](08_gpt_desde_cero.ipynb) | ~16 min |
| 09 | [`09_vision_mas_alla_de_la_clasificacion.ipynb`](09_vision_mas_alla_de_la_clasificacion.ipynb) | ~6 min |

## Cómo empezar

Requisitos: Python 3.10 o superior.

```bash
git clone https://github.com/fborbon/academy.git
cd academy/neural-networks/es
python3 -m venv .venv && source .venv/bin/activate
pip install -r ../requirements.txt
jupyter lab
```

Abre los notebooks en orden. Ejecútalos de uno en uno: varios núcleos de PyTorch en paralelo compiten por los mismos hilos de la CPU y cada uno puede ir muchas veces más lento.

## Sobre la traducción

Estos notebooks se generan con los mismos scripts que la edición en inglés, y el código es idéntico: solo se traducen las explicaciones, los comentarios y las etiquetas de las gráficas, así que ambas ediciones producen los mismos resultados. Dos conjuntos de datos se quedan en inglés a propósito, y cada notebook explica por qué: el corpus de embeddings del notebook 06 y el corpus de entrenamiento del GPT del notebook 08. La documentación técnica completa (estructura del repositorio, despliegue y costes) está en el [README principal](../README.md), en inglés.

## Contacto

Preguntas o sugerencias: forwardforecasting@gmail.com

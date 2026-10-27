# Need4Speed

Need4Speed es un proyecto desarrollado en el marco del [Hackatón Momentum](https://aprende.csic.es/catalogo/hackaton-momentum/), organizado por CSIC Aprende.
Es una experiencia intensiva de aprendizaje e innovación colaborativa en Granada.
El reto se centra en diseñar y entrenar modelos de Aprendizaje por Refuerzo Profundo (DRL).
En este proyecto desarrollamos agentes para entornos de simulación mediante trabajo en equipo.

# Contenido

## Material didáctico

En el directorio `ejemplos/` están disponibles los siguientes _notebooks_:

- `01_entornos.ipynb`: material de introducción a la librería `gymnasium` y el entorno `highway_env`.
- `02_agentes.ipynb`: definición de agentes con `stable-baselines3`, uso de _callbacks_ y optimización de parámetros con `optuna`.
- `03_wrappers.ipynb`: uso de _wrappers_ y herramientas avanzadas para la resolución del problema propuesto.

##  _Notebooks_ entregables

- `test.ipynb`: contiene el código empleado para ejecutar los mejores agentes obtenidos.
- `train.ipynb`: contiene el código empleado para entrenar los agentes entregados.

Al final de `test.ipynb` se explica qué hay que entregar: una carpeta con el nombre del equipo, el `best_model.zip` y, si se usaron wrappers, un `wrappers.py`.

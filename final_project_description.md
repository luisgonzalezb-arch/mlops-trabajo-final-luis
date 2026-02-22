# MLOps: Proyecto Final

## A) Definición del problema (Problem Definition)

### Caso de uso (AI/ML Use Case)
Este proyecto implementa un caso de **regresión supervisada** cuyo objetivo es **predecir la progresión de la diabetes** (salida numérica) a partir de 10 variables de entrada estandarizadas. El modelo entrenado se expone mediante una **API REST** para realizar inferencias.

### Contexto, objetivo y beneficios
En contextos reales, modelos predictivos pueden apoyar el seguimiento de pacientes y la priorización de casos. En este trabajo (de enfoque académico), el objetivo es **demostrar el ciclo de vida de un sistema de ML** con prácticas de MLOps: adquisición de datos, preparación, entrenamiento, evaluación con métricas, serialización del modelo y despliegue para inferencia.

**Resultado esperado:** recibir un conjunto de *features* y devolver una **predicción numérica** consumible por API.

### Restricciones y supuestos
- El dataset es **pequeño** y público; el desempeño está limitado por el tamaño de muestra.
- Las variables se encuentran **estandarizadas** (propio del dataset de referencia), por lo que no se realiza una “limpieza” intensiva; se priorizan **validaciones de calidad** y trazabilidad del flujo.
- El proyecto no busca uso clínico real, sino evidenciar competencias de ML/MLOps a nivel académico.

### Métrica de éxito (Success Metric)
La métrica principal es **RMSE (Root Mean Squared Error)** sobre un conjunto de prueba (*hold-out*).

- **Criterio de éxito (alto nivel):** obtener un RMSE menor que un baseline simple (por ejemplo, regresión lineal) y dejar evidencia reproducible de métricas/experimentos en el repositorio.


## Data Acquisition

### 1) Identificar (fuente del dataset)
Para este proyecto se utiliza el dataset público **Diabetes** provisto por **scikit-learn**, accesible mediante `sklearn.datasets.load_diabetes`. Se emplea como dataset de referencia académica para un problema de **regresión supervisada**.

### 2) Definir (qué es el dataset y qué representa)
- **Tipo de problema:** Regresión (predicción de un valor numérico).
- **Unidad de análisis:** 1 observación corresponde a 1 registro/paciente (según la definición del dataset).
- **Objetivo (target):** `y`, variable continua a predecir asociada a la progresión/medida definida por el dataset.
- **Entradas (features):** 10 variables numéricas estandarizadas.

### 3) Describir (estructura y variables)
- **Features (10):** `age`, `sex`, `bmi`, `bp`, `s1`, `s2`, `s3`, `s4`, `s5`, `s6`
- **Target:** `y`

### 4) Adquirir y almacenar (raw → processed)
La adquisición del dataset se realiza desde scikit-learn y se exporta a un archivo CSV en la carpeta `data/raw/`.  
Posteriormente, se genera un dataset de entrenamiento en la carpeta `data/training/` como resultado del proceso de preparación de datos.

Artefactos generados dentro del repositorio:
- **Dataset raw:** `data/raw/diabetes.csv`
- **Dataset de entrenamiento (processed):** `data/training/processed.csv`

### 5) Analizar (análisis básico del dataset raw)
Para “analizar” el dataset raw antes del entrenamiento, se consideran los siguientes controles básicos (data quality checks):

- **Dimensión del dataset:** verificación del número de filas y columnas.
- **Valores faltantes (nulos):** validación de nulos por columna y total.
- **Duplicados:** verificación de filas duplicadas.
- **Tipos de datos:** revisión de tipos por columna (numéricos esperados).
- **Consistencia de variables:** confirmación de la existencia del target (`y`) y de las 10 features esperadas.

**Nota:** el dataset Diabetes de scikit-learn se entrega típicamente estandarizado y sin valores faltantes; por ello, el foco de esta etapa es documentar adecuadamente la adquisición, validar consistencia básica y asegurar trazabilidad del flujo `raw → processed`.

## C) ML Experimentation (Experimentación y Evaluación)

La experimentación, evaluación y selección del modelo campeón se documenta en el siguiente notebook (Google Colab):

- **Notebook (Colab):** https://colab.research.google.com/drive/11Y0YkhtecAujJ93O5f8p4khgKV7yb9fw

### Evidencias incluidas en el notebook
- Ejecución de experimentos en Jupyter Notebook.
- Data Preparation con diagnóstico (nulos, duplicados, outliers y correlación) y justificación de tratamiento.
- Entrenamiento y comparación de múltiples modelos de clasificación (baseline y candidatos).
- Evaluación con métricas (ROC-AUC como métrica principal, además de F1/Recall/Accuracy).
- Selección del modelo campeón (mejor ROC-AUC) y soporte con:
  - Curva ROC
  - Matriz de confusión
  - Reporte por clase (classification_report) y recall por clase (malignant/benign).
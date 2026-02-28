# MLOps: Proyecto Final

## A) Definición del problema (Problem Definition)

### Caso de uso (AI/ML Use Case)
Este proyecto implementa un caso de **clasificación binaria supervisada** cuyo objetivo es **predecir si un tumor es maligno o benigno** a partir de **30 características numéricas** calculadas sobre imágenes de núcleos celulares (Breast Cancer Wisconsin - Diagnostic).  
El modelo entrenado se expone mediante una **API REST** para realizar inferencias.

### Contexto, objetivo y beneficios
En contextos reales, modelos de clasificación pueden apoyar el análisis y priorización de casos (**no reemplazan el diagnóstico clínico**).  
En este trabajo (enfoque académico), el objetivo es demostrar el ciclo de vida completo de un sistema de ML con prácticas de MLOps: **adquisición de datos, preparación, entrenamiento, evaluación con métricas, serialización del modelo y despliegue para inferencia**.

**Resultado esperado:** recibir un conjunto de *features* y devolver una **predicción de clase** (malignant/benign) junto con **probabilidades** consumibles por API.

### Restricciones y supuestos
- El dataset es público y de tamaño limitado; el desempeño está acotado por la muestra.
- El proyecto no busca uso clínico real, sino evidenciar competencias de ML/MLOps a nivel académico.
- La selección del modelo se basa en métricas sobre un split **train/test** reproducible.

### Métrica de éxito (Success Metric)
- **Métrica principal:** **ROC-AUC** (mayor es mejor).
- **Métricas complementarias:** **F1**, **Recall por clase** (malignant y benign) y **Accuracy**.
- **Criterio de éxito (alto nivel):** obtener un ROC-AUC alto y evidenciar evaluación reproducible; reportar desempeño por clase considerando el mapeo:
  - **0 = malignant**
  - **1 = benign**

---

## B) Data Acquisition (Adquisición de datos)

### 1) Identificar (fuente del dataset)
Para este proyecto se utiliza el dataset público **Breast Cancer Wisconsin (Diagnostic)** provisto por **scikit-learn**, accesible mediante `sklearn.datasets.load_breast_cancer`.

### 2) Definir (qué es el dataset y qué representa)
- **Tipo de problema:** Clasificación binaria (0/1).
- **Unidad de análisis:** 1 observación corresponde a un caso/paciente (según definición del dataset).
- **Objetivo (target):** `target`, con mapeo:
  - 0 = malignant
  - 1 = benign
- **Entradas (features):** 30 variables numéricas (medidas “mean”, “error” y “worst”).

### 3) Describir (estructura y variables)
- **N features:** 30
- **Features:**
  - mean radius, mean texture, mean perimeter, mean area, mean smoothness,
  mean compactness, mean concavity, mean concave points, mean symmetry, mean fractal dimension,
  radius error, texture error, perimeter error, area error, smoothness error,
  compactness error, concavity error, concave points error, symmetry error, fractal dimension error,
  worst radius, worst texture, worst perimeter, worst area, worst smoothness,
  worst compactness, worst concavity, worst concave points, worst symmetry, worst fractal dimension
- **Target:** `target` (0=malignant, 1=benign)

### 4) Adquirir y almacenar (raw → processed)
La adquisición del dataset se realiza desde scikit-learn y se exporta a un archivo CSV en la carpeta `data/raw/`. Posteriormente, se genera un dataset de entrenamiento en la carpeta `data/training/` como resultado del proceso de preparación de datos.

Artefactos generados dentro del repositorio:
- **Dataset raw:** `data/raw/breast_cancer.csv` *(o el nombre real que uses en tu pipeline)*
- **Dataset de entrenamiento (processed):** `data/training/processed.csv`

### 5) Analizar (análisis básico del dataset raw)
Se consideran controles básicos (data quality checks):
- Dimensión del dataset (filas/columnas)
- Valores faltantes (nulos)
- Duplicados
- Tipos de datos esperados (numéricos)
- Consistencia del target y features esperadas

---

## C) ML Experimentation (Experimentación y Evaluación)

La experimentación, evaluación y selección del modelo campeón se documenta en el notebook (Google Colab):
- **Notebook (Colab):** https://colab.research.google.com/drive/11Y0YkhtecAujJ93O5f8p4khgKV7yb9fw

### Evidencias incluidas en el notebook
- Ejecución de experimentos en Jupyter Notebook.
- Data Preparation con diagnóstico (nulos, duplicados, outliers y correlación) y justificación de tratamiento.
- Comparación de modelos (baseline y candidatos).
- Evaluación con métricas:
  - ROC-AUC (principal)
  - F1, Recall por clase (malignant/benign), Accuracy
- Soporte con:
  - Curva ROC
  - Matriz de confusión
  - classification_report

### Modelo campeón (Champion)
- **Modelo:** LogisticRegression + StandardScaler
- **Test size:** 0.20
- **random_state:** 42
- **threshold:** 0.50
- **Resultados (test):**
  - ROC-AUC: 0.9953703703703703
  - F1: 0.9861111111111112
  - Recall (benign=1): 0.9861111111111112
  - Recall (malignant=0): 0.9761904761904762
  - Accuracy: 0.9824561403508771

---

## D) ML Development (Desarrollo del modelo)

### Preparación de datos
- Export `raw` a `data/raw/`
- Construcción de dataset de entrenamiento en `data/training/processed.csv`
- Separación reproducible train/test (test_size=0.2, random_state=42)
- Estandarización con StandardScaler (para el modelo lineal)

### Entrenamiento y serialización
- Entrenamiento en `src/train.py`
- Serialización del modelo en `models/model.joblib`
- Lógica de inferencia en `src/predict.py`
- Tests en `tests/test_predict.py`

---

## E) Serving (Despliegue/Servicio)

Se implementa un servicio local vía API REST (Flask) en `src/serving.py`.

### Endpoint: /predict
- **Input:** JSON con las 30 features (nombres exactos)
- **Output:** clase predicha y probabilidades (benign/malignant), manteniendo el mapeo 0/1

---

## F) Conclusiones, limitaciones y mejoras futuras

### Conclusiones
- Se construyó un flujo end-to-end reproducible (data → train → evaluate → serialize → serve).
- LogisticRegression + StandardScaler logró desempeño sobresaliente (ROC-AUC ~0.995) con buena sensibilidad para la clase malignant.

### Limitaciones
- Dataset académico y público: no representa variabilidad clínica real.
- No se incluye monitoreo de drift ni validación externa (poblaciones/hospitales distintos).
- Serving local: no se implementa despliegue productivo (Docker/Cloud) ni CI/CD.

### Mejoras futuras
- MLflow para tracking de experimentos y model registry.
- Docker + CI/CD (GitHub Actions) para automatizar tests/build/deploy.
- Validación de schema de entrada (pydantic) y manejo robusto de errores.
- Monitoreo de latencia, tasa de errores y drift (data/model).

### Lecciones aprendidas
- La documentación y trazabilidad del pipeline es clave para reproducibilidad.
- Mantener consistente el mapeo de clases (0/1) evita errores en inferencia y métricas.
- Separar lógica de predicción del serving facilita testing y mantenimiento.

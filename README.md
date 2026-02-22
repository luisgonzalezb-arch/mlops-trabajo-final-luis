# MLOps Introduction: Final Project

Final work description in the final_project_description.md file.

## Student info
- Full name: [LUIS SANITAGO GONZALEZ BRAVO]
- e-mail: [luis.gonzalez.b@uni.pe]
- Grupo: [-]
- Project Name: Breast Cancer Wisconsin — Clasificación (Benigno vs Maligno)

---

## A) Problem Definition

### AI/ML Use Case
Este proyecto implementa un caso de **clasificación binaria supervisada** para predecir si un tumor es **maligno** o **benigno** a partir de características numéricas calculadas sobre imágenes de núcleos celulares.

### Contexto, objetivo y beneficios
En un contexto médico real, un modelo de clasificación puede servir como apoyo para el análisis y priorización de casos (no reemplaza diagnóstico clínico).  
En este trabajo (enfoque académico), el objetivo es demostrar el ciclo de vida completo de un sistema de ML/MLOps: adquisición/preparación de datos, entrenamiento, evaluación con métricas, serialización del modelo y despliegue para inferencia vía API.

### Restricciones y supuestos
- Dataset público, tamaño limitado, orientado a fines académicos.
- El modelo y resultados no deben interpretarse como herramienta clínica.
- La selección del modelo se basa en métricas sobre un split de train/test.

### Métrica de éxito (alto nivel)
- Métrica principal: **ROC-AUC** (mayor es mejor).
- Métricas complementarias: **F1**, **Recall**, **Accuracy**.
- Criterio de éxito: lograr ROC-AUC alto y evidenciar evaluación reproducible; en particular, reportar desempeño por clase considerando el mapeo `0 = malignant`, `1 = benign`.

---

## Data Acquisition (dataset raw)

### Fuente del dataset
Se utiliza el dataset público **Breast Cancer Wisconsin (Diagnostic)** provisto por **scikit-learn** mediante `load_breast_cancer`.

### Variables
- Features: 30 variables numéricas (medidas “mean”, “error” y “worst”).
- Target: `target`, con mapeo:
  - `0 = malignant`
  - `1 = benign`

### Artefactos del proyecto
- Dataset raw: `data/raw/` (archivo CSV exportado por el pipeline)
- Dataset procesado/entrenamiento: `data/training/processed.csv`

---

## B) Project Preparation
- Repositorio público en GitHub.
- Estructura del proyecto organizada (carpetas `data/`, `src/`, `models/`, `reports/`, `tests/`).
- Desarrollo realizado en ramas secundarias y consolidado mediante Pull Requests hacia `main`.

---

## C) ML Experimentation

La experimentación, evaluación y selección del modelo campeón se documenta en el siguiente notebook:

- Google Colab: https://colab.research.google.com/drive/11Y0YkhtecAujJ93O5f8p4khgKV7yb9fw

### Evidencias incluidas en el notebook
- Execution: Experimentos ejecutados en Jupyter Notebook.
- Data Preparation: diagnóstico de nulos, duplicados, outliers y correlación; experimentos “con vs sin tratamiento” para justificar decisiones.
- Model Selection: comparación de múltiples modelos (baseline y candidatos).
- Evaluation: evaluación con ROC-AUC, F1, Recall y Accuracy; análisis con curva ROC, matriz de confusión y `classification_report`.
- Champion: selección del modelo campeón por mayor ROC-AUC.

---

## D) ML Development Activities

### Data Preparation
- Dataset raw guardado en `data/raw/`.
- Transformaciones/estructura para dataset de entrenamiento guardadas en `data/training/processed.csv`.
- Features y target definidos de forma consistente en el pipeline.

### Model Training Implementation
- Entrenamiento implementado en `src/train.py`.
- Modelo serializado en `models/` (por ejemplo `models/model.joblib`).
- Métricas registradas en `reports/` (cuando aplica).

---

## E) Model Deployment & Serving
- Se implementa un servicio local vía API REST (Flask) en `src/serving.py`.
- El endpoint `/predict` permite enviar features y recibir una predicción (clase y/o probabilidad).

---

## F) Delivery
- Este `README.md` centraliza la documentación del trabajo.
- Se incluyen scripts (`src/`), tests (`tests/`) y artefactos (`models/`, `reports/`, `data/`).
- La entrega final se consolida en la rama `main` mediante Pull Requests.


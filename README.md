# MLOps: Proyecto Final

La descripción del trabajo se encuentra en: `final_project_description.md`.

## Información del estudiante
- **Nombre completo:** Luis Santiago Gonzalez Bravo
- **Correo:** [luis.gonzalez.b@uni.pe]
- **Grupo:** [-]
- **Nombre del proyecto:** API de predicción (Regresión) con dataset público (Diabetes - scikit-learn)

## Descripción breve del proyecto
Este proyecto implementa un flujo MLOps:
1) Preparación de datos (dataset público).
2) Entrenamiento y serialización de un modelo.
3) Servicio de predicción mediante una API REST (`/predict`).

## Dataset

### Raw dataset
- **Fuente:** Dataset público **Diabetes** incluido en `scikit-learn` (`sklearn.datasets.load_diabetes`).
- **Archivo generado en el repositorio:** `data/raw/diabetes.csv`

### Training dataset
- **Generado por:** `src/data_preparation.py`
- **Archivo:** `data/training/processed.csv`

## Artefacto del modelo
- **Modelo entrenado:** `models/model.joblib` (generado por `src/train.py`)

## Serving (API)
- **Script:** `src/serving.py`
- **Endpoints:**
  - `GET /` (health check)
  - `POST /predict` (predicción)

## Evidencia de funcionamiento (ejemplo)

Entrada usada (POST `/predict`):
```json
{"age":0.05,"sex":0.02,"bmi":0.04,"bp":0.01,"s1":0.03,"s2":-0.02,"s3":-0.01,"s4":0.02,"s5":0.04,"s6":0.01}
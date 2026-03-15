# CreditVision

CreditVision es una aplicación de **evaluación de riesgo crediticio** desarrollada con **Streamlit** que utiliza una **Red Neuronal Artificial (ANN)** combinada con **reducción de dimensionalidad mediante PCA** para clasificar el perfil financiero de un cliente en tres categorías de riesgo.

La aplicación permite ingresar variables financieras reales del cliente y obtener una **predicción del Credit Score en tiempo real**.

---

## Objetivo del proyecto

El objetivo del proyecto es construir un modelo de **Machine Learning supervisado** capaz de predecir la calidad del historial crediticio de un cliente a partir de variables financieras y comportamentales.

El sistema clasifica el perfil en tres categorías:

- **Bad** — Alto riesgo crediticio  
- **Standard** — Riesgo medio  
- **Good** — Bajo riesgo crediticio  

---

## Tecnologías utilizadas

El proyecto fue desarrollado utilizando:

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- PCA (Principal Component Analysis)
- Joblib

---

## Metodología del modelo

El flujo completo del modelo es el siguiente:

1. Limpieza y depuración de datos  
2. Codificación de variables categóricas mediante `LabelEncoder`  
3. Normalización de variables numéricas con `MinMaxScaler`  
4. Reducción de dimensionalidad mediante **PCA (8 componentes)**  
5. Entrenamiento de una **Red Neuronal Artificial (ANN)**  
6. Evaluación del modelo en conjunto de prueba  
7. Implementación del modelo en una aplicación web con **Streamlit**

---

## Resultados del modelo

Evaluación realizada sobre el **conjunto de prueba**.

### Reporte de clasificación

| Clase | Precision | Recall | F1-Score | Support |
|------|-----------|--------|---------|--------|
| Bad (0) | 0.97 | 0.99 | 0.98 | 624 |
| Standard (1) | 0.97 | 0.98 | 0.97 | 917 |
| Good (2) | 0.99 | 0.93 | 0.96 | 334 |

### Métricas globales

| Métrica | Valor |
|--------|------|
| Accuracy | **0.9739 (97.39%)** |
| Loss | 0.0801 |
| Macro Avg Precision | 0.98 |
| Macro Avg Recall | 0.97 |
| Macro Avg F1-Score | 0.97 |
| Weighted Avg Precision | 0.97 |
| Weighted Avg Recall | 0.97 |
| Weighted Avg F1-Score | 0.97 |

El modelo presenta **alto desempeño predictivo**, logrando identificar correctamente los perfiles de riesgo crediticio con una precisión superior al **97%**.

---

## Aplicación Streamlit

La aplicación permite ingresar información financiera del cliente como:

- número de cuentas bancarias
- número de tarjetas de crédito
- tasa de interés
- número de préstamos
- retrasos en pagos
- número de pagos atrasados
- cambio en el límite de crédito
- consultas de crédito
- deuda pendiente
- utilización del crédito
- antigüedad del historial crediticio
- cuota mensual total
- balance mensual
- mezcla de crédito
- pago del monto mínimo

A partir de estos datos, la aplicación genera:

- clasificación del riesgo crediticio
- nivel de confianza del modelo
- probabilidad para cada categoría

---

## Estructura del repositorio


credit-score-app-v2/

app.py
requirements.txt
README.md

modelo_riesgo_credito.keras
minmax_scaler.joblib
pca_8_componentes.joblib
label_encoders.joblib

notebooks/
└── Riesgo.ipynb


### Descripción de archivos

| Archivo | Descripción |
|------|------|
| app.py | Aplicación Streamlit para predicción |
| modelo_riesgo_credito.keras | Red neuronal entrenada |
| minmax_scaler.joblib | Escalador de variables |
| pca_8_componentes.joblib | Reducción de dimensionalidad |
| label_encoders.joblib | Codificación de variables categóricas |
| requirements.txt | Dependencias del proyecto |
| notebooks/Riesgo.ipynb | Notebook de entrenamiento del modelo |

---

## Ejecutar el proyecto localmente

Clonar el repositorio:

```bash
git clone https://github.com/leydymf/credit-score-app-v2.git
cd credit-score-app-v2
``` 

Crear entorno virtual:

```bash
python -m venv venv
source venv/bin/activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar la aplicación:

```bash
streamlit run app.py
```

La aplicación se abrirá en:

```bash
http://localhost:8501
```

Dataset

El dataset utilizado contiene información financiera de clientes como:

ingresos

deuda

comportamiento de pago

historial crediticio

utilización de crédito

Fuente del dataset:

```bash
https://github.com/adiacla/bigdata/raw/master/riesgo.xlsx
```

Autor

Leydy Yohana Macareo Fuentes

Ingeniería de Sistemas
Ciencia de Datos

Licencia

Este proyecto fue desarrollado con fines académicos.


---

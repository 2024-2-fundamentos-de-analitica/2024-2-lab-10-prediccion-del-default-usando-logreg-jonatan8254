import os
import gzip
import json
import pickle
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix

# Función para preparar y limpiar los datos
def preprocess_dataframe(df: pd.DataFrame):
    df_clean = df.copy()
    df_clean = df_clean.rename(columns={"default payment next month": "default"})
    df_clean.drop(columns=["ID"], inplace=True)
    df_clean.dropna(inplace=True)
    df_clean = df_clean[(df_clean["EDUCATION"] != 0) & (df_clean["MARRIAGE"] != 0)]
    df_clean["EDUCATION"] = df_clean["EDUCATION"].apply(lambda v: 4 if v >= 4 else v).astype("category")
    X = df_clean.drop(columns=["default"])
    y = df_clean["default"]
    return df_clean, X, y

# Creación del pipeline de procesamiento y modelo
def build_model_pipeline() -> Pipeline:
    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
    transformer = ColumnTransformer(
        transformers=[
            ("cat_enc", OneHotEncoder(), categorical_features)
        ],
        remainder=MinMaxScaler()
    )
    pipe = Pipeline([
        ("transformer", transformer),
        ("selector", SelectKBest(score_func=f_regression)),  # k se optimiza posteriormente
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
    return pipe

# Optimización de hiperparámetros con GridSearchCV, ampliando el rango de C
def tune_hyperparameters(pipeline_obj, features, target):
    # Ajustar el preprocesador para conocer la dimensión final de las características
    transformer = pipeline_obj.named_steps["transformer"]
    X_trans = transformer.fit_transform(features)
    num_features = X_trans.shape[1]
    
    param_grid = {
        "selector__k": range(1, num_features + 1),
        "clf__C": [0.1, 1, 10, 100, 1000],
        "clf__solver": ["liblinear", "lbfgs"]
    }
    
    grid_cv = GridSearchCV(
        estimator=pipeline_obj,
        param_grid=param_grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        verbose=2,
        refit=True
    )
    grid_cv.fit(features, target)
    return grid_cv

# Guardado del modelo optimizado en formato comprimido
def persist_model(model_obj):
    os.makedirs("files/models", exist_ok=True)
    with gzip.open("files/models/model.pkl.gz", "wb") as out_file:
        pickle.dump(model_obj, out_file)

# Función para determinar el mejor umbral basándose en balanced_accuracy
def choose_best_threshold(model, X, y):
    proba = model.predict_proba(X)[:, 1]
    best_threshold = 0.5
    best_score = 0
    for t in np.linspace(0, 1, 101):
        preds = (proba >= t).astype(int)
        score = balanced_accuracy_score(y, preds)
        if score > best_score:
            best_score = score
            best_threshold = t
    return best_threshold

# Función de predicción personalizada usando un umbral determinado
def custom_predict(model, X, threshold):
    proba = model.predict_proba(X)[:, 1]
    return (proba >= threshold).astype(int)

# Cálculo de métricas utilizando la predicción con umbral personalizado
def calculate_metrics(model_obj, X_train, y_train, X_test, y_test, threshold):
    pred_train = custom_predict(model_obj, X_train, threshold)
    pred_test = custom_predict(model_obj, X_test, threshold)
    
    metrics_train = {
        "type": "metrics",
        "dataset": "train",
        "precision": float(precision_score(y_train, pred_train)),
        "balanced_accuracy": float(balanced_accuracy_score(y_train, pred_train)),
        "recall": float(recall_score(y_train, pred_train)),
        "f1_score": float(f1_score(y_train, pred_train))
    }
    
    metrics_test = {
        "type": "metrics",
        "dataset": "test",
        "precision": float(precision_score(y_test, pred_test)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred_test)),
        "recall": float(recall_score(y_test, pred_test)),
        "f1_score": float(f1_score(y_test, pred_test))
    }
    return metrics_train, metrics_test

# Cálculo de la matriz de confusión usando las predicciones con umbral personalizado
def compute_confusion_matrix(model_obj, X_train, y_train, X_test, y_test, threshold):
    pred_train = custom_predict(model_obj, X_train, threshold)
    pred_test = custom_predict(model_obj, X_test, threshold)
    
    cm_train = confusion_matrix(y_train, pred_train)
    cm_test = confusion_matrix(y_test, pred_test)
    
    conf_train = {
        "type": "cm_matrix",
        "dataset": "train",
        "true_0": {"predicted_0": int(cm_train[0, 0]), "predicted_1": int(cm_train[0, 1])},
        "true_1": {"predicted_0": int(cm_train[1, 0]), "predicted_1": int(cm_train[1, 1])}
    }
    
    conf_test = {
        "type": "cm_matrix",
        "dataset": "test",
        "true_0": {"predicted_0": int(cm_test[0, 0]), "predicted_1": int(cm_test[0, 1])},
        "true_1": {"predicted_0": int(cm_test[1, 0]), "predicted_1": int(cm_test[1, 1])}
    }
    
    return conf_train, conf_test

# Escritura de los resultados en un archivo JSON
def write_metrics_to_file(train_metrics, test_metrics, cm_train, cm_test, filepath="files/output/metrics.json"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    results = [train_metrics, test_metrics, cm_train, cm_test]
    with open(filepath, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

# --- Flujo principal del script ---

data_test = pd.read_csv("files/input/test_data.csv.zip", compression="zip")
data_train = pd.read_csv("files/input/train_data.csv.zip", compression="zip")

data_test, X_test, y_test = preprocess_dataframe(data_test)
data_train, X_train, y_train = preprocess_dataframe(data_train)

model_pipe = build_model_pipeline()
best_model = tune_hyperparameters(model_pipe, X_train, y_train)
persist_model(best_model)

# Calcular el umbral óptimo en el conjunto de entrenamiento
threshold = choose_best_threshold(best_model, X_train, y_train)

metrics_train, metrics_test = calculate_metrics(best_model, X_train, y_train, X_test, y_test, threshold)
conf_train, conf_test = compute_confusion_matrix(best_model, X_train, y_train, X_test, y_test, threshold)
write_metrics_to_file(metrics_train, metrics_test, conf_train, conf_test)

# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

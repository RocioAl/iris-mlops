# Clasificación de Flores Iris — Práctica de Git

Proyecto de Machine Learning para clasificar especies de flores Iris (`Iris-setosa`, `Iris-versicolor`, `Iris-virginica`) a partir de las medidas de sus sépalos y pétalos.

## Configuración del proyecto
- **Dataset:** `data/Iris.csv` (150 registros, 4 características numéricas).
- **Modelo principal:** Regresión Logística

## Estructura del repositorio
- `data/Iris.csv`: Conjunto de datos original.
- `src/train_models.py`: Script principal que entrena y evalúa los clasificadores.
- `requirements.txt`: Dependencias necesarias para ejecutar el proyecto.

## Ejecución
```bash
pip install -r requirements.txt
python src/train_models.py
```

Este proyecto utiliza Regresión Logística, Random Forest y SVM.

<<<<<<< HEAD
## Evaluación de modelos
Evaluación de modelos
=======
#Modelo SVM
Es la maquina de vector soporte
>>>>>>> feature/svm

## Accuracy
Mide qué tan cerca está un resultado, dato o medición del valor real o verdadero

## Precision 
 Es una métrica de evaluación para modelos de clasificación que mide la calidad de las predicciones positivas realizadas por el modelo

 ## Recall
 mide la capacidad de un modelo de clasificación para identificar todos los casos positivos reales.

 ## F1-score
 Es una métrica utilizada en Machine Learning para evaluar el rendimiento de un modelo de clasificación
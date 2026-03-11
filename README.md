
# Predicción de Enfermedad Cardiovascular con ML

Este proyecto compara dos modelos de **Machine Learning** para predecir la presencia de enfermedad cardiovascular a partir de datos clínicos.

El objetivo es aplicar un flujo completo de **preparación de datos, entrenamiento de modelos y evaluación de rendimiento** para determinar cuál de los modelos ofrece mejores resultados en la clasificación.

## Modelos evaluados

- **Regresión Logística**
- **Random Forest**

---

## Dataset

El dataset utilizado (`heart_modified.csv`) contiene variables clínicas de pacientes, entre ellas:

- **age** → edad  
- **sex** → sexo  
- **chol** → nivel de colesterol  
- **trestbps** → presión arterial en reposo  
- otras variables clínicas  

Variable objetivo:

- **output**
  - `0` → no enfermedad cardiovascular
  - `1` → presencia de enfermedad cardiovascular

---

## Flujo de trabajo

1. Carga y análisis inicial del dataset  
2. Limpieza de datos  
   - Imputación de valores nulos  
   - Eliminación de registros duplicados  
3. Detección y eliminación de **outliers** mediante IQR  
4. Selección de variables relevantes (`age`, `sex`, `chol`)    
5. División del dataset (80% entrenamiento / 20% prueba)  
6. Escalado de variables numéricas para **Regresión Logística**
7. Entrenamiento de modelos  
8. Optimización de hiperparámetros con **GridSearchCV**  
9. Evaluación del rendimiento  
10. Predicción de nuevos casos introducidos por el usuario  

---

## Resultados de los modelos

| Modelo | Precisión |
|--------|-----------|
| Regresión Logística | **0.7368** |
| Random Forest | 0.603 |

La **Regresión Logística** obtuvo mejor rendimiento en este dataset.

---

## Uso del script

Ejecutar el programa desde consola:

```bash
python prediccion_cardiovascular.py
```

El script entrenará los modelos y permitirá introducir nuevos datos para realizar una predicción.

## Ejemplo de entrada
```
Edad: 55
Sexo: M
Colesterol: 220
```
## Ejemplo de salida
```
Predicción: Paciente tiene enfermedad cardiovascular
Probabilidad: 0.73
```

---
  
## Tecnologías utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn

---

## Autor

IvanJFor

"""
@author: IvanJFor

EJERCICIO MACHINE LEARNING - PREDICCIÓN CARDIOVASCULAR

Este script carga un dataset clínico para entrenar dos modelos de Machine Learning
(Regresión Logística y Random Forest) para predecir si un paciente tiene riesgo de
enfermedad cardiovascular en base a su edad, género y colesterol.
"""

# -- LIBRERÍAS NECESARIAS --

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


# -- CARGA DATOS --
print('\n-- CARGA, ANÁLISIS Y PREPARACIÓN DE LOS DATOS --')

# Se cargan los datos desde un CSV y se convierten a un DataFrame
df = pd.read_csv('heart_modified.csv')

# Se estudia el contenido del DataFrame por pantalla
print('\nDataFrame:')
print(df)
print('\nColumnas:')
print(df.columns)
print('\nShape:')
print(df.shape)
print('\nDescribe:')
print(df.describe())
print('\nHead:')
print(df.head())


# -- 1. VALORES FALTANTES --

# Se comprueban las que tienen datos nulos para luego rellenarlos con la media y la moda
if df.isnull().values.any():
    nulos = df.isnull().sum()
    nulos = nulos[nulos > 0]
    print('\nColumnas con valores nulos:')
    print(nulos)

    # Se rellena la edad con la Media redondeada sin decimales
    media_edad = round(df['age'].mean())
    print('\nSe procede a rellenar la edad con la Media:', media_edad)
    df['age'] = df['age'].fillna(media_edad)

    # El resto con la Moda (valores que más se repiten)
    moda_todos = df.mode().iloc[0]
    print('\nEl resto se rellenará con la Moda de cada una...')
    print('\nModa de cada columna:')
    print(moda_todos)
    df = df.fillna(moda_todos)

# Se vuelve a comprobar si existen datos nulos
nulos = df.isnull().sum()
nulos = nulos[nulos > 0]
print('\nColumnas con valores nulos:')
print(nulos)


# -- 2. FILAS DUPLICADAS --

# Se detecta si hay filas duplicadas para en tal caso eliminarlas
if df.duplicated().any():
    duplicados = df.duplicated().sum()
    print('\nFilas duplicadas:\n', duplicados)

    # Eliminar filas duplicadas
    print('\nSe eliminan las filas duplicadas...')
    df = df.drop_duplicates()

# Se comprueba si quedan todavía
if df.duplicated().any():
    print('\nFilas duplicadas:\n', df.duplicated().sum())
else:
    print('\nNo quedan filas duplicadas:', df.shape)


# -- 4. REDUCCIÓN DATAFRAME --

# Se reduce el DataFrame a las columnas útiles
print('\nSe reduce el DataFrame a las columnas útiles [edad, sexo, colesterol y resultado]')
df = df[['age', 'sex', 'chol', 'output']]
print('\nDataFrame Reducido:\n')
print(df)


# -- 3. OUTLIERS --

# Se escogen las columnas numéricas para realizar la búsqueda de Outliers
col_numericas = df.select_dtypes(include="number").columns
Q1 = df[col_numericas].quantile(0.25)
Q3 = df[col_numericas].quantile(0.75)

# Se establece el rango Intercuartílico de los campos numéricos (podrían tenerlos 'age' y 'chol')
print('\nSe establece el rango Intercuartílico para obtener los valores atípicos...')
IQR = Q3 - Q1
print('\nQ1:')
print(Q1)
print('\nQ3:')
print(Q3)
print('\nIQR:')
print(IQR)

# Y con él se definen los límites superior e inferior
lim_inferior = Q1 - 1.5 * IQR
lim_superior = Q3 + 1.5 * IQR
print('\nLímite Inferior:')
print(lim_inferior)
print('Límite Superior:')
print(lim_superior)

# Se obtienen los Outliers
outliers = (df[col_numericas] < lim_inferior) | (df[col_numericas] > lim_superior)
print('\nOutliers:')
print(df[outliers.any(axis=1)])

# Se eliminan los Outliers
print('\nSe eliminan los Outliers...')
# Si solo se hubiera hecho con la edad
#df = df[(df['age'] >= lim_inferior) & (df['age'] <= lim_superior)]
# Se crea una máscara para solo quedarnos con los datos del rango correcto
mascara = ~outliers.any(axis=1) # equivalente a np.logical_not(outliers.any(axis=1))
df = df[mascara].copy()
print('\nDataFrame sin Outliers:\n')
print(df)
print(df.shape)

# Identificamos el rango de colesterol que queda para las pruebas (126-360)
print('\nRango colesterol:' , df['chol'].min(), '-', df['chol'].max())

# No es necesario convertir la variable categórica 'sex' en numérica por ya venir así
#print("\nSe convierte la columna categórica 'sex' a numérica...")
#df['sex'] = df['sex'].map({'male': 0, 'female': 1})

print('\nDataFrame final:')
print(df)


# -- 4 SEPARACIÓN DE CARACTERÍSTICAS Y ETIQUETAS --
# Lo normal sería hacerlo después del escalado, pero de esta manera optimizamos al descartarlos antes

# Se identifican las variables independientes (X) y dependientes (y)
X = df[['age', 'sex', 'chol']]
y = df['output']


# -- 5. ESCALADO DE DATOS NUMÉRICOS --

# (Para que la diferencia de rango entre ellas no les otorgue más peso a unas que a otras)
print('\nEscalado de datos...')
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[['age', 'chol']] = scaler.fit_transform(X_scaled[['age', 'chol']])


# -- 6 DIVISIÓN DEL CONJUNTO DE DATOS --

# Se dividen los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
# stratify=y se usa para mantener la proporción de la columna y si no estuviera bien balanceada


# -- 7. REGRESIÓN LOGÍSTICA --

print('\n--- MODELO DE REGRESIÓN LOGÍSTICA ---')
print('\nEspere por favor...')

# Se crea el modelo
modelo_RL = LogisticRegression(max_iter=1000, random_state=42)

# Se aplica evaluación por Validación Cruzada
scores_log = cross_val_score(modelo_RL, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
print(f'\nRegresión Logística Precisión Validación Cruzada: {scores_log.mean():.3f} + {scores_log.std()}')

# Ajuste final y test
modelo_RL.fit(X_train, y_train)
y_pred_RL = modelo_RL.predict(X_test)
precision_RL = accuracy_score(y_test, y_pred_RL)

# Se muestran los resultados
print(f'\nRegresión Logística Precisión Test: {precision_RL:.4f}')

# -- 8. RANDOM FOREST --

print('\n--- MODELO DE RANDOM FOREST ---')
print('\nEspere por favor...')

# Se crea el modelo de Random Forest
modelo_RF = RandomForestClassifier(random_state=42)

# Ajuste de hiperparámetros cond GridSearchCV
param_grid_RF = {
    'n_estimators': [100, 200, 300], # n_estimators indica el número de árboles en el bosque aleatorio.
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2]
}

# Se entrena el modelo (no haría falta usar las variables escaladas, pero tampoco influye que lo estén)
#modelo_RF.fit(X_train, y_train) # se podría omitir porque GridSearchCV ya entrena también

# Búsqueda de hiperparámetros con validación cruzada
grid_RF = GridSearchCV(modelo_RF, param_grid_RF, cv=5, n_jobs=-1, scoring='accuracy')
grid_RF.fit(X_train, y_train)

# Se evalúa el modelo
print("Mejores parámetros RF:", grid_RF.best_params_)
print(f"Random Forest Validación Cruzada Mejor Puntuación: {grid_RF.best_score_:.3f}")

# -- 9. EVALUACIÓN DE LOS MODELOS --

print('\nEvaluación de los modelos...\n')

# Se hacen las predicciones
best_RF = grid_RF.best_estimator_  # Este sería el mejor modelo RF después del CV
y_pred_RF = best_RF.predict(X_test)
precision_RF = accuracy_score(y_test, y_pred_RF)
print(f"Random Forest Precisión Test: {precision_RF:.4f}")

# Comparación del mejor modelo
if precision_RL > precision_RF:
    print('\nEl mejor modelo es Regresión Logística con una precisión de:', precision_RL )
    mejor_modelo = modelo_RL
elif precision_RL < precision_RF:
    print('\nEl mejor modelo es Random Forest con una precisión de:', precision_RF)
    mejor_modelo = best_RF # modelo optimizado
else:
    print('\nAmbos modelos tienen la misma precisión:', precision_RF)
    mejor_modelo = best_RF

# Se prueba el modelo con datos solicitados al usuario
print('\n--- SIMULACIÓN CON NUEVOS DATOS ---\n')
edad = int(input("Introduzca la edad del paciente: "))
sexo = input("Introduzca el sexo del paciente (M ó F): ") # Masculino (0) - Femenino (1)
colesterol = int(input("Introduzca el nivel del colesterol del paciente (100-400): "))


df_nuevo = pd.DataFrame([
    {'age': edad,
     'sex': 0 if sexo.upper()=='M' else 1,
     'chol': colesterol
     }
])

# Se escalan los datos introducidos con el escalador entrenado
df_nuevo[['age', 'chol']]= scaler.transform((df_nuevo[['age', 'chol']]))

# Se predice prueba
prediccion = mejor_modelo.predict(df_nuevo)[0]
probabilidad = mejor_modelo.predict_proba(df_nuevo)[0]

# Mostrar resultado
print(f"\nPredicción: {'Paciente tiene Enfermedad Cardiovascular' if prediccion==1 else 'Paciente NO tiene enfermedad'}")
print(f"\nProbabilidades: No tener enfermedad {probabilidad[0]:0.2f}, Tener Enfermedad Cardiovascular {probabilidad[1]:0.2f}")
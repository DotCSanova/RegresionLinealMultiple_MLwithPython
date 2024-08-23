#En primer lugar, al igual que con regresión lineal simple y con el resto, necesitamos
#importar las librerías de las que vayamos hacer uso

#region Librerías
import matplotlib.pyplot as plt #Para representación de gráficas
import numpy as np #Para realizar cálculos numéricos eficientes con datos (nos da los ndarray por ej.)
import pandas as pd #Ofrece el objeto DataFrame, que nos permite trabajar con datos tabulares enriquecidos con etiquetas en filas y columnas.
import pylab as pl #Junta en un mismo espacio de nombres funciones de MatplotLib y Numpy, asemejándose a Matlab
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D
#endregion

#region Descarga datos

#El siguiente paso siempre es descargar el archivo de datos (normalmente .csv). En este caso ya lo tenemos descargado,
# sin emgargo, el comando para ello es el siguiente:

# !curl -o FuelConsumption.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv

#La herramienta curl o "Client URL" es una herramienta de línea de comandos que sirve para verificar conectiviad a URL y transferir datos.
#En esta script se incluye un ! delante para indicar en python que se trata de una herramienta de línea de comandos.
#En este caso, al escribir -o, nos permite indicar el nombre de archivo a ponerle cuando se guarde y/o indicarle una ubicación diferente al directorio local.
#Otra forma es escribiendo -O, en cuyo caso se guardará el archivo en el directorio de trabajo actual con el mismo nombre de archivo que el remoto.

# !curl -O https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv

#endregion

#region Visualización y comprensión de la data

data = pd.read_csv("FuelConsumption.csv") #Para leer el csv y convertirlo a un dataframe con Pandas

print(data.head(9)) #Visualizar las 9 primeras filas empezando desde arriba. Al no ser un cuaderno Jupyter y querer ver la salida de head() por consola hay que incluir el print.

#Con este comando vemos varios datos: Tenemos 13 columnas o características (modelyear, make, model,...).
#En este caso vamos a tomar CO2EMISSIONS como variable dependiente, ya que es la que queremos predecir, y tomaremos como variables independientes
#las siguientes características: 'ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB'.

# !! Importante destacar que añadir más variables independientes sin ninguna base teórica puede llevar a overfitting -> más no es mejor.

#Por tanto, creamos a continuación un df diferente extrayendo únicamente las características que nos interesan:

extracted_data = data[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']] #Seleccionar únicamente las columnas que nos interesan del dataframe 'data'

print(extracted_data.head(9))

extracted_data2 = data['ENGINESIZE'] #Esto sería una serie. Los dataframes son objetos bidimensionales (filas x columnas) mientras que las series son objetos unidimensionales con datos indexados. Se puede entender como que una columna de un df es una serie. Los Df además tienen títulos en cada columna.

print(extracted_data2.head(9))

#En los problemas de regresión lineal es importante asegurarnos de que existe relación lineal entre la (s) variables independientes y la variable dependiente a predecir. El mejor método para comprobar la dependencia lineal es haciendo uso de scatter plot para verificar visualmente esta linealidad.

#Comprobación linealidad de ENGINESIZE VS CO2EMISSIONS mediante análisis gráfico.

plt.scatter(extracted_data.ENGINESIZE, extracted_data.CO2EMISSIONS, color='blue')
plt.xlabel('Engine size')
plt.ylabel('CO2 emissions')
plt.show()

#Comprobación lienalidad de CYLINDERS VS CO2EMISSIONS mediante análisis gráfico.

plt.scatter(extracted_data.CYLINDERS, extracted_data.CO2EMISSIONS, color='red')
plt.xlabel('Number of cylinders')
plt.ylabel('CO2 Emissions')
plt.show()

#Comprobación linealidad FUELCONSUMPTIONCITY VS CO2EMISSIONS mediante análisis gráfico

plt.scatter(extracted_data.FUELCONSUMPTION_CITY, extracted_data.CO2EMISSIONS, color='green')
plt.xlabel('FuelConsumption in the city')
plt.ylabel('CO2 Emissions')
plt.show()

#endregion

#region División de data en train y test

#La división de la data en train y test split es muy importante ya que nos permite conocer de forma fiel el comportamiento del modelo fuera de la muestra (out-of-sample accuracy).
#Dado que se divide la data en train para entrenar el modelo y test para testear, el modelo no conoce los datos del set de test, por lo que a la hora de evaluar es perfecto ya que 
#conocemos los resultados de este set y para el modelo son datos fuera de la muestra.

#Conozco dos formas de hacer el train/test split, presento ambas a continuación:

#region Método 1 train/test split

#mask = np.random.rand(len(extracted_data)) < 0.8 #Primero creamos una máscara. Se crea unn array de números aletorios entre 0 y 1 con la longitud del dataframe de datos extraídos. Luego, se pone la condición de  <0.8 ya que vamos a coger 80% del dataset para entrenamiento y el resto (20)para test

#La variable mask es un array de booleanos (true o false) según se cumpla la condición de que los números aleatorios generados sean < 0.8 o > 0.8.

#train = extracted_data[mask] #Aplico la máscara al set de datos para que de esta forma, todas las filas que sean TRUE se guarden formando el set de train
#test = extracted_data[~mask] #Aplico la máscara al set de datos para que de esta forma, todas las filas que sean FALSE se guarden formando el set de test

#!!IMPORTANTE!! -> Cuando aplicas una máscara booleana a un DataFrame en Pandas, el resultado es un nuevo DF que contiene solo las filas donde la máscara es true (o false si se pide así como en el set test)

#!! IMPORTANTE!! -> A destacar que con este método no estamos realmente cogiendo el 80% y 20% del set de datos para crear el tain set y test set respectivamente, ya que los números generados son aleatorios y no tiene por qué haber una distribución uniforme de 80% números que cumplan la condición < 0.8

#endregion

#region Metodo 2 train/test split de Scikit-learn

#Sklearn nos aporta un paquete o funcionalidad que nos permite hacer fácil y rápidamente la separación en train/test

X = extracted_data[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']] #Resultado -> DF
y = extracted_data['CO2EMISSIONS'] #Resultado -> Serie

X_train, X_test, y_train, y_test = train_test_split( X,
                                                    y,
                                                    train_size=0.8,
                                                    shuffle = True,
    
)

#Al separar de este modo observamos como la división del 80/20% se cumple exactamente.

#endregion
#endregion

#region Creación y entrenamiento del modelo

#Vamos a crear el modelo en base a la separación del método 2.

#Para crear un modelo de regresión lineal hay que importar previamente el paquete LinearRegresion de Sklearn.

modelo = linear_model.LinearRegression()
modelo.fit(X_train, y_train)


#Si hemos separado previamente los datos con el Método 1, el entrenamiento se hace de la siguiente forma

#modelo = linear_model.LinearRegression()
#x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
#y = np.asanyarray(train[['CO2EMISSIONS']])
#modelo.fit (x, y)

#En este caso vemos como se transforman los dataframes en arrays. No es extrictamente necesario ya que sklearn puede trabajar con Dataframes.
#Esto es ya que a de hace esta transformación de forma interna. Sin embargo, aunque Sklearn acepta Dataframes de Pandas, dado que de forma interna los 
#transforma a arrays de numpy, algunos desarrolladores prefieren hacerlo de forma previa para asegurarse de que los datos están en el formato esperado.

#endregion

#region Imprimir Coeff y Intercep

print("Coeficiente:", [(col, float(coef)) for col, coef in zip(X.columns, modelo.coef_.flatten())]) #De esta forma se muestra una tupla de Columna (variable indep) con su coeficiente.
print("Intercept o Sesgo:",modelo.intercept_)

#La intersección o intercept es el valor theta0, mientras que los coeficientes son el resto de theta que acompañan a a los X (variables indep).
#Estos valores son los que conforman el hiperplano de la regresión lineal múltiple, Sklearn los estima a partir de los datos mediante el método
#OLS (Ordinary Least Squares). 
# 
# !!! IMPORTANTE TEORÍA -> OLS (Ordinary Least Squares) intenta minimizar la suma de los errores cuadrados (SSE)
# o el error cuadrático medio (MSE) entre la variable objetivo (y) y nuestra salida predicha (y hat) a lo largo de todas las muestras en el conjunto de datos.
# Esto se puede conseguir mediante dos enfoques:
#   1)Resolver los parámetros del modelo analíticamente mediante operaciones de álgebra lineal. 
#   2)Usando algoritmos de optimización (por ejemplo, el Descenso de gradiente)

#endregion

#region Predicción de variable dependiente.

#Con Método 1 de separación:

y_pred = modelo.predict(X_test) #Predecimos la variable dependiente CO2EMISSIONS para las filas del DF de X_test (que contiene las variable independiente) con el modelo previamente entrenado. El resultado es un arreglo o array unidimensional (214,). La coma al final indica que es un arreglo unidimensional, con 214 elementos.
print(y_pred) #Imprimimos en pantalla los valores predichos de CO2EMISSIONS
print(y_pred.shape) #Imprimimos en pantalla la dimensión del array, por motivos de entender correctamente lo que devuelve la función predict().

#Con Método 2 de separación:

#y_pred= modelo.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])

#endregion

#region Imprimir hiperplano

#Imprimir ecuación del plano

print('Ecuación del plano: y = {}*ENGINE SIZE + {}*CYLINDERS + {}*FUELCONSUMPTION_CITY + {}*FUELCONSUMPTION_HWY + {} '.format(round(modelo.coef_[0],3), round(modelo.coef_[1],3), round(modelo.coef_[2],3), round(modelo.coef_[3],3), round(modelo.intercept_,3)))

#Al tener 4 variables independientes es complejo graficar el plano en 3D, haría falta procesos de reducción de dimesnionalidad.
# Probar a hacerlo con otro modelo de dos variables independientes. Referencia: https://www.youtube.com/watch?v=aWRypv0teBc 

#endregion

#region Datos de rendimiento del modelo. MSE y R^2.

#MSE 

print("Mean Squared Error (MSE) : %.2f" % np.mean((y_pred - y_test) ** 2))

#R^2 o variance score, 1 is perfect prediction

print('Variance score: %.2f' % modelo.score(X_test, y_test))


#endregion


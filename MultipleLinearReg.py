#En primer lugar, al igual que con regresión lineal simple y con el resto, necesitamos
#importar las librerías de las que vayamos hacer uso

#region Librerías
import matplotlib.pyplot as plt #Para representación de gráficas
import numpy as np #Para realizar cálculos numéricos eficientes con datos (nos da los ndarray por ej.)
import pandas as pd #Ofrece el objeto DataFrame, que nos permite trabajar con datos tabulares enriquecidos con etiquetas en filas y columnas.
import pylab as pl #Junta en un mismo espacio de nombres funciones de MatplotLib y Numpy, asemejándose a Matlab
from sklearn.model_selection import train_test_split
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

X = extracted_data[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']]
Y = extracted_data['CO2EMISSIONS']

X_train, X_test, y_train, y_test = train_test_split( X,
                                                    Y.values.reshape(-1,1),
                                                    train_size=0.8,
                                                    shuffle = True,
    
)

print(X_train)
#endregion


#endregion
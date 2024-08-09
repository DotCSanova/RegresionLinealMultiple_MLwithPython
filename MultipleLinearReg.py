#En primer lugar, al igual que con regresión lineal simple y con el resto, necesitamos
#importar las librerías de las que vayamos hacer uso

#region Librerías
import matplotlib.pyplot as plt #Para representación de gráficas
import numpy as np #Para realizar cálculos numéricos eficientes con datos (nos da los ndarray por ej.)
import pandas as pd #Ofrece el objeto DataFrame, que nos permite trabajar con datos tabulares enriquecidos con etiquetas en filas y columnas.
import pylab as pl #Junta en un mismo espacio de nombres funciones de MatplotLib y Numpy, asemejándose a Matlab
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
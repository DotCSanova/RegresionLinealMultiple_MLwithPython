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
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 17:29:06 2022

@author: selecto
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
from sklearn import linear_model


#----Descarga de el archivo de datos-------------------------------------
site_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv'
r = requests.get(site_url, allow_redirects=True)
open('FuelConsumption.csv', 'wb').write(r.content)
#------------------------------------------------------------------------

#--------Creacion del dataframe con los datos descargados----------------
df = pd.read_csv("FuelConsumption.csv")

#-------Extraigo los datos de interes de todo el dataframe---------------
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

#-----Grafico tamano de motor y co2 en puntos-----------------------------
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

#------Separo los datos de el dataframe de forma aleatoria en 80% y 20%---
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]#-------------------Estos datos son de entrenamiento(20%)
test = cdf[~msk]#-------------------Estos datos son los de prueba del modelo (80%)

#-------Grafico los datos de entrenamiento de motor y co2-----------------
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='orange')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()



#----Modelo----------------------------------------------------------

regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x, y)
# The coefficients
print ('Coeficientes: ', regr.coef_)


#-----Prediccion----------------------------------------------------

y_hat= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print("Residuo de la suma de los cuadrados: %.2f"
      % np.mean((y_hat - y) ** 2))



# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))

#-------Grafica de los tres factores y la linea ajustada-------------
fig,ax = plt.subplots()
p = plt.subplot(1, 1, 1)
p.set_xlabel("Engine size")
p.set_ylabel("Emission")
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='orange')
plt.scatter(train.CYLINDERS, train.CO2EMISSIONS,  color='red')
plt.scatter(train.FUELCONSUMPTION_COMB, train.CO2EMISSIONS,  color='green')

xT=x.transpose()
x2=regr.coef_[0][0]*xT[0] + regr.coef_[0][1]*xT[1] + regr.coef_[0][2]*xT[2]
x3=x2.transpose()

plt.plot(x,  x3 + regr.intercept_[0], '-b')
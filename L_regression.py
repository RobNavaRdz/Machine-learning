# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 13:02:34 2022

@author: Roberto Nava
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
from sklearn import linear_model
from sklearn.metrics import r2_score


#----Descarga de el archivo de datos-------------------------------------
site_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv'
r = requests.get(site_url, allow_redirects=True)
open('FuelConsumption.csv', 'wb').write(r.content)
#------------------------------------------------------------------------

#--------Creacion del dataframe con los datos descargados----------------
df = pd.read_csv("FuelConsumption.csv")

#--------Muestra los valores estadisticos del dataframe------------------
print(df.describe())

#-------Extraigo los datos de interes de todo el dataframe---------------
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print(cdf.head(9))
#-------De los datos extraidos los ordeno a conveniencia y creo un histograma
viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()
#-----Grafico los datos de combustible y co2 por puntos-------------------
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='green')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()
#-----Grafico tamano de motor y co2 en puntos-----------------------------
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='red')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()
#-----Grafico los cilindros contra co2------------------------------------
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Cylinders")
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



#-------Defino el modelo a utilizar-------------------------------------------
regr = linear_model.LinearRegression()#---------regresion lineal
train_x = np.asanyarray(train[['ENGINESIZE']])#-----elijo los datos de x
train_y = np.asanyarray(train[['CO2EMISSIONS']])#---elijo los datos de y
regr.fit(train_x, train_y)#-------Hago que el modelo se ajuste con x e y dados


#-----------Muestro los coeficientes del modelo-------------------------------
print ('Coefficients: ', regr.coef_)#--------esto es a, o la pendiente
print ('Intercept: ',regr.intercept_)#-------esto es b, o la interceccion


#---Grafico por puntos el motor y co2, ademas de la recta ajustada del modelo
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")

#----Convierto a arreglo los datos de prueba para x e y-----------------------
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)#----realiza el calculo de valores para estas x
print(test_y_)
#-----Imprimo los valores de medicion del error del modelo--------------------
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_) )

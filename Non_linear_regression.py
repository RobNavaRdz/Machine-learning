# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 16:29:48 2022

@author: selecto
"""

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


def sigmoid(x, Beta_1, Beta_2):
     y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
     return y

#----Descarga del archivo csv desde IBM -----------------------------
site_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/china_gdp.csv'
r = requests.get(site_url, allow_redirects=True)
open('china_gdp.csv', 'wb').write(r.content)

#-----Ordenamos los datos en un dataframe-------------------------------
df = pd.read_csv("china_gdp.csv")
print(df.head(10))

#-----Analizamos los datos viendo el comportamiento graficamente------------
plt.figure(figsize=(8,5))
x_data, y_data = (df["Year"].values, df["Value"].values)
plt.plot(x_data, y_data, 'ro')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

#---Creamos una curva logistica con valore beta aleatorios------------------
beta_1 = 0.10
beta_2 = 1990.0

#logistic function
Y_pred = sigmoid(x_data, beta_1 , beta_2)

#plot initial prediction against datapoints
plt.plot(x_data, Y_pred*15000000000000.)
plt.plot(x_data, y_data, 'ro')

# Lets normalize our data
xdata =x_data/max(x_data)
ydata =y_data/max(y_data)

#---usando el ajuste de curva de scipy calculamos las betas-----------------
popt, pcov = curve_fit(sigmoid, xdata, ydata)
#print the final parameters
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))

#---Graficamos el resultado de la curva logistica ajustada con los datos---
x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

#---Para hacer el analisis de los datos creamos una mascara para test y entrenamiento
msk=np.random.random(len(df))<0.8
train_x=xdata[msk]
train_y=ydata[msk]
test_x=xdata[~msk]
test_y=ydata[~msk]

#---Ajustamos la curva logistica con los datos de entrenamiento
popt, pcov = curve_fit(sigmoid, train_x, train_y)
#---Obtenemos la curva con los datos de test------------------------------
y_=sigmoid(test_x, *popt)

#----Calculamos los parametros de ajuste de la curva-----------------------
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y,y_ ) )
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:40:07 2022

@author: Roberto Nava
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import r2_score


def yvalues (coeficients, xvalues):
    """Funcion que genera el polinomio ajustado al grado indicado"""
    Yval=0
    for i in range (1,len(coeficients[0]+1)):
        Yval+=coeficients[0][i]*(xvalues**i)
    return Yval

#----Descarga del archivo csv desde IBM -----------------------------
site_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv'
r = requests.get(site_url, allow_redirects=True)
open('FuelConsumption.csv', 'wb').write(r.content)

#----Creación del dataframe y la vista de los primeros 5 renglones---
df = pd.read_csv("FuelConsumption.csv")
print(df.head())

#-----Del dataframe original extraemos la información de interés-----
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print(cdf.head(9))

#---Graficamos los valores de emisión contra tamaño de motor---------
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

#---De los datos originales tomamos el 80% para entrenar el modelo----
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

#--Convertimos los datos en arreglos numpy --------------------------
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

#--Definimos el ajuste polinomial al grado deseado-----------------
poly = PolynomialFeatures(degree=3)
train_x_poly = poly.fit_transform(train_x)
print(train_x_poly)

#---Con la matriz de valores de x tratamos como regresion lineal multiple
clf = linear_model.LinearRegression()
train_y_ = clf.fit(train_x_poly, train_y)

#---Con coeficientes-----------------------------------------------
print ('Coefficients: ', clf.coef_)
print ('Intercept: ',clf.intercept_)

#---Graficamos los datos y el polinomio obtenido-------------------
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
#--Creamos un arreglo de 0 a 10 en pasos de 0.1 como tamaños de motor
XX = np.arange(0.0, 10.0, 0.1)
#--Los valores de y es el valor de intercepcion y los valores de la funcion yvalues
yy = clf.intercept_[0]+ yvalues(clf.coef_, XX)
plt.plot(XX, yy, '-r' )
plt.xlabel("Engine size")
plt.ylabel("Emission")
        
#---Usamos los datos de test para obtener los valores de ajuste----
test_x_poly = poly.transform(test_x)
test_y_ = clf.predict(test_x_poly)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y,test_y_ ) )

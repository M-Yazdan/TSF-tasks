# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 08:02:11 2020

@author: myazd
"""

"""TSF TASK#1"""
'''Predict the percentage of a student based on the no. of study hours.'''
'''Muhammad Yazdan'''


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



dataset = pd.read_csv('score.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.4, random_state=0)


#regression
from sklearn.linear_model import LinearRegression
reg= LinearRegression()
reg.fit(x_train,y_train)

#predict
y_predict= reg.predict(x_test)

from sklearn.metrics import r2_score

r2=print(r2_score(y_test, y_predict))

sample = np.array([[9.25]])
predicted_score=print(reg.predict(sample))


plt.scatter(x_train,y_train,color='red')
plt.plot(x_train, reg.predict(x_train))
plt.title('Linear Reg Hours VS Scores')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()


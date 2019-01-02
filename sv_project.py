'''
Support Vector Machine Project
Iris Flower Data Set
'''
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

iris = sns.load_dataset('iris')
print(iris.info())

'''
Exploratory Data Analysis
'''

sns.pairplot(iris, hue = 'species')
plt.show()

setosa = iris[iris['species']=='setosa']
sns.jointplot('sepal_length', 'sepal_width', data = setosa, kind = 'kde')
plt.show()

'''
Train Test Split
'''
from sklearn.model_selection import train_test_split
X= iris.drop('species', axis = 1) 
y = iris['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

'''
Train a model 
'''

from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)

'''
Model Evaluation
'''
predictions = model.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))


'''
Grid Search
'''
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1,10,100,1000], 'gamma': [0.1,0.01,0.001,0.0001]}

grid = GridSearchCV(SVC(), param_grid, verbose=3)
grid.fit(X_train, y_train)

grid_predictions = grid.predict(X_test)

print(classification_report(y_test, grid_predictions))
print('\n')
print(confusion_matrix(y_test, grid_predictions))





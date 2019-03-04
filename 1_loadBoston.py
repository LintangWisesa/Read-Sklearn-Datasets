import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

dataBoston = load_boston()

# show data from load_boston
# print(dataBoston)

# show key properties dict dataBoston
# print(dir(dataBoston))

# print(dataBoston['data'])
# print(len(dataBoston['data']))
# print(dataBoston['feature_names'])
# print(dataBoston['data'][0])
# print(dataBoston['target'][0])

# ===============================

# create dataframe
boston = pd.DataFrame(
    dataBoston['data'], 
    columns = dataBoston['feature_names']
)
boston['MEDV'] = dataBoston['target']
# print(boston)

# =============================

from sklearn import linear_model
model = linear_model.LinearRegression()

# training
model.fit(boston[
    ['CRIM', 'ZN', 'INDUS', 'CHAS',
    'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
    'B','LSTAT']
], boston['MEDV'])

# coeffisient slope m
# print(model.coef_)

# intercept b
# print(model.intercept_)

# accuracy
# print(model.score(boston[
#     ['CRIM', 'ZN', 'INDUS', 'CHAS',
#     'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
#     'B','LSTAT']
# ], boston['MEDV']))

# prediction
print(model.predict([[
    0.02731,0,7.07,0,0.469,6.421,78.9,
    4.9671,2,242,17.8,396.9,9.14
]]))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

dataDiabet = load_diabetes()
print(dir(dataDiabet))

# print(dataDiabet['feature_names'])
# print(dataDiabet['data'][0])
# print(dataDiabet['target'])

# ===================================

diabet = pd.DataFrame(
    dataDiabet['data'],
    columns = dataDiabet['feature_names']
)
diabet['gula_darah'] = dataDiabet['target']
# print(diabet)

# ===============================

from sklearn import linear_model
print(dataDiabet['feature_names'])
model = linear_model.LinearRegression()

# training
model.fit(diabet[['age', 'sex', 'bmi', 
    'bp', 's1', 's2', 
    's3', 's4', 's5', 
    's6']],
    diabet['gula_darah']
)

# nilai slope m
print(model.coef_)

# nilai intercept b
print(model.intercept_)

# prediksi
print(model.predict([[0.038076, 0.05068, 0.061696,
    0.021872, -0.044223, -0.034821, -0.043401,
    -0.002592, 0.019908, -0.017646  ]]
))

# akurasi
print(model.score(diabet[['age', 'sex', 'bmi', 
    'bp', 's1', 's2', 
    's3', 's4', 's5', 
    's6']],
    diabet['gula_darah'])
)
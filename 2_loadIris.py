import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

dataIris = load_iris()
# print(dataIris)

# print(dataIris['feature_names'])
# print(dataIris['data'][0])
# print(dataIris['target'])
# print(dataIris['target_names'])

# ================================

# create dataframe
iris = pd.DataFrame(
    dataIris['data'],
    columns = dataIris['feature_names']
)
iris['target'] = dataIris['target']
iris['spesies'] = iris['target'].apply(
    lambda x: dataIris['target_names'][x]
)
# print(iris)

# ==================================

# separate dataframe by species
setosa = iris[iris['spesies']=='setosa']
# print(setosa)
versicolor = iris[iris['spesies']=='versicolor']
# print(versicolor)
virginica = iris[iris['spesies']=='virginica']
print(virginica)

# ================================
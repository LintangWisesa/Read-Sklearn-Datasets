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
print(diabet)
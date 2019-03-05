import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
digits = load_digits()

user = input('Ketik tanggal lahir : ')
count = 1

fig = plt.figure('Digit', figsize=(12,2))
for x in user:
    plt.subplot(1,6,count)
    plt.imshow(digits['images'][int(x)], cmap='gray')
    plt.title('Angka = {}'.format(digits['target'][(int(x))]))
    count += 1

plt.show()
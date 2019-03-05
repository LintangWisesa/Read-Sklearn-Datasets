import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()

print(dir(digits))
# print(digits['data'][0])
# print(digits['images'][13])
# print(digits['target'][13])
# print(digits['target_names'][13])

# ===============================
# plot 1 digit

# print(digits['target'][1796])
# fig = plt.figure('Digit', figsize=(6,6))
# plt.imshow(digits['images'][1796], cmap='gray_r')
# plt.show()

# ============================
# plot 3 digit

# fig = plt.figure('Digit', figsize=(9,3))

# plt.subplot(131)
# plt.imshow(digits['images'][0], cmap='gray')
# plt.title('Angka = {}'.format(digits['target'][0]))
# plt.subplot(132)
# plt.imshow(digits['images'][1], cmap='gray')
# plt.title('Angka = {}'.format(digits['target'][1]))
# plt.subplot(133)
# plt.imshow(digits['images'][2], cmap='gray')
# plt.title('Angka = {}'.format(digits['target'][2]))

# plt.show()

# ==================================
# plot 10 digit w/ for loop

fig = plt.figure('Digit', figsize=(12,4))

for x in range(10):
    plt.subplot(2,5,x+1)
    plt.imshow(digits['images'][x], cmap='gray')
    plt.title('Angka = {}'.format(digits['target'][x]))

plt.show()
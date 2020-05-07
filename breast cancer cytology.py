"""

faye

"""
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.preprocessing.image
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.metrics
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.tree
import sklearn.ensemble
import os
import datetime
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm


x_images = np.load("C:/Users/Faye/Downloads/love.npy/X.npy")
# load labels of shape (5547,1); (0 = no cancer, 1 = cancer)
y_images = np.load("C:/Users/Faye/Downloads/faye.npy")

# shuffle data
perm_array = np.arange(len(x_images))
np.random.shuffle(perm_array)
x_images = x_images[perm_array]
y_images = y_images[perm_array]

print("x_images.shape =", x_images.shape)
print(
    "x_images.min/mean/std/max = %.2f/%.2f/%.2f/%.2f"
    % (x_images.min(), x_images.mean(), x_images.std(), x_images.max())
)
print("")
print("y_images.shape =", y_images.shape)
print(
    "y_images.min/mean/std/max = %.2f/%.2f/%.2f/%.2f"
    % (y_images.min(), y_images.mean(), y_images.std(), y_images.max())
)

imgs_0 = x_images[y_images == 0]  # 0 = no cancer
imgs_1 = x_images[y_images == 1]  # 1 = cancer

plt.figure(figsize=(20, 20))
for i in range(30):
    plt.subplot(5, 6, i + 1)
    plt.title("IDC = %d" % y_images[i])
    plt.imshow(x_images[i])

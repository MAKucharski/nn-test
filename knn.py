import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D, Conv2D, MaxPool2D
import numpy as np
#import matplotlib.pyplot as plt
import os
import random
import xlrd
import sklearn
from numpy import unique
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn import preprocessing
from sklearn import metrics
import time
import datetime
from tensorflow.keras.callbacks import TensorBoard

datadir = "C:\\Users\\Mateusz\\Desktop\\dataset"
categories = ["01. one", "02. two", "03. three", "04. four", "05. five", "06. six", "07. seven",
              "08. eight", "09. nine"] # zmiana w wersji 1.1; zmiana w wersji 1.2

training_data = []


def create_training_data():
    for category in categories:
        path = os.path.join(datadir, category)
        class_num = categories.index(category)
        for xlsx in os.listdir(path):
            book = xlrd.open_workbook(os.path.join(path, xlsx))
            sheet = book.sheet_by_name('Sheet1') # zmiana w 1.1
            txt_array = [[sheet.cell_value(r, c) for c in range(sheet.ncols)] for r in range(sheet.nrows)]
            txt_array = np.array(txt_array)
            training_data.append([txt_array, class_num])



create_training_data()

random.shuffle(training_data)
#for sample in training_data:
    #print(sample[1])
X = []  # data
Y = []  # label

for features, label in training_data:
    X.append(features)
    Y.append(label)

X = np.array(X).reshape(-1, 300)
print(X.shape)
#print(X)
Y = np.array(Y)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
print(len(X_train))
print(X_train.shape)
print(len(y_train))
print(len(X_test))
print(X.shape)
#zeby dzialalo usunac X = X.reshape(22, 4, 1)
knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

expect = y_test
prediction = knn.predict((X_test))
print(prediction)
print(metrics.classification_report(expect, prediction))
print(knn.score(X_test, y_test))
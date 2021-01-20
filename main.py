import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D, Conv2D, MaxPool2D
import numpy as np
#import matplotlib.pyplot as plt
import os
import random
import xlrd
from numpy import unique
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import time
import datetime
from tensorflow.keras.callbacks import TensorBoard

NAME = "formanty-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
'''
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")
'''

datadir = "C:\\Users\\Mateusz\\Desktop\\dataset"
categories = ["01. one", "02. two", "03. three", "04. four", "05. five", "06. six", "07. seven",
              "08. eight", "09. nine"] # zmiana w wersji 1.1; zmiana w wersji 1.2
'''

for category in categories:
    path = os.path.join(datadir, category)
    for txt in os.listdir(path):
        txt_array = pd.read_csv(os.path.join(path,txt), sep ="\t")
        #print(txt_array)
'''


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

X = np.array(X).reshape(-1, 100, 3)
print(X.shape)
#print(X)
Y = np.array(Y)
#print(type(Y))
#print(Y)
#X = tf.keras.utils.normalize(X)
#print(X)
#print(X.shape)
#X = X.reshape([1446, 100, 3]) # zmiana w wersji 1.1
#print(X.shape)
#print(X)
#X = X.reshape(1446, 100, 3)
#print(X.shape)
#print(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.001)
print(len(X_train))
print(X_train.shape)
print(len(y_train))
print(len(X_test))
print(X.shape)
''' #zeby dzialalo usunac X = X.reshape(22, 4, 1)
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)
X_new = np.array([[632.610377, 1443.110758, 2376.79294, 3404.070208]])
X_new = tf.keras.utils.normalize(X_new)
prediction = knn.predict((X_new))
X_new2 = np.array([[387.371016,	1423.05739,	2334.429814, 3262.155058]])
X_new2 = tf.keras.utils.normalize(X_new2)
prediction2 = knn.predict((X_new2))
print(prediction, prediction2)
print(knn.score(X_test, y_test))
'''


model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=X_train.shape[1:])) #zmiana w 1.1
model.add(Dense(16, activation='relu'))
model.add(MaxPooling1D())

model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D())

model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D())

model.add(Flatten())
model.add(Dense(9, activation='softmax')) #zmiana 1.2 3->9


model.compile(loss="sparse_categorical_crossentropy", optimizer="adam",
              metrics=['accuracy'])
model.summary()
tf.keras.utils.plot_model(
    model, to_file='model.png', show_shapes=False,
    show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96
) #nowe 1.2

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(X_train, y_train, batch_size=20, epochs=30, validation_split=0.3, callbacks=[tensorboard_callback])
# 1.2 usunięcie verbose=0, dodano validation_split=0.3
acc = model.evaluate(X_train, y_train)
print("Loss: ", acc[0], " Accuracy: ", acc[1])

'''
model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
#model.add(MaxPooling1D)

model.add(Flatten)
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
model.summary()

model.fit(X, Y, batch_size=32, epochs=3, validation_split=0.1)
'''

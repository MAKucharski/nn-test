import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D
import numpy as np
#import matplotlib.pyplot as plt
import os
import random
import xlrd
from numpy import unique
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
'''
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")
'''

datadir = "C:\\Users\\Mateusz\\Desktop\\dataset"
categories = ["a", "e"]
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
            sheet = book.sheet_by_name('Arkusz1')
            txt_array = [[sheet.cell_value(r, c) for c in range(sheet.ncols)] for r in range(sheet.nrows)]
            txt_array = np.array(txt_array)
            print(type(txt_array))
            #txt_array.to_numpy()
            print(txt_array)
            #txt_array.select_dtypes(include=float).to_numpy()
            training_data.append([txt_array, class_num])
            #print(type(training_data))
            #print(training_data)


create_training_data()

random.shuffle(training_data)
# for sample in training_data:
    # print(sample[1])
X = []  # data
Y = []  # label

for features, label in training_data:
    X.append(features)
    Y.append(label)

# X = X.float() ponoć ma być float potem?
#print(X.shape) # dlaczego features jest 1? powinno być 4
#X = X.reshape(X.shape[0], X.shape[1], 1)
Y = np.array(Y)
print(type(Y))
print(Y)
print(type(X))
print(X)
X = tf.keras.utils.normalize(X)
print(X)
print(X.shape)
X = X.reshape([22, 4])
print(X.shape)
print(X)
X = X.reshape(22, 4, 1)
print(X.shape)
print(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)
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
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(4, 1)))
model.add(Dense(16, activation='relu'))
#model.add(Activation("relu"))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(3, activation='softmax'))
#model.add(Activation('softmax'))

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam",
              metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, batch_size=16, epochs=3, verbose=0)
acc = model.evaluate(X_train, y_train)
print("Loss: ", acc[0], " Accuracy: ", acc[1])

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D
import numpy as np
#import matplotlib.pyplot as plt
import os
import random
from numpy import unique
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
        for txt in os.listdir(path):
            txt_array = pd.read_csv(os.path.join(path, txt), sep="\t")
            training_data.append([txt_array, class_num])
            # print(type(txt_array))


create_training_data()
# (len(training_data))
# print(training_data)

random.shuffle(training_data)

# for sample in training_data:
    # print(sample[1])
X = []  # data
Y = []  # label

for features, label in training_data:
    X.append(features)
    Y.append(label)

print(type(X))

X = pd.DataFrame(X).to_numpy()
# X = X.float() ponoć ma być float potem?
print(type(X))
print(X.shape) # dlaczego features jest 1? powinno być 4
print(X)
X = X.reshape(X.shape[0], X.shape[1], 1)
print(X.shape)
print(X)
X = tf.keras.utils.normalize(X)

model = Sequential()
#model.add(Conv1D(64, (3,1), input_shape = X.shape[1:]))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(22, 4)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
#model.add(Activation("relu"))
model.add(MaxPooling1D(pool_size=(2)))

#model.add(Conv1D(64, (3,1)))
model.add(Activation("relu"))
model.add(MaxPooling1D(pool_size=(2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('softmax'))

model.compile(loss="binary_crossentropy", optimizer="adam",
              metrics=['accuracy'])

model.fit(X, Y, batch_size=32, epochs=3)

#a_train = pd.read_csv("C:/Users/48784/Desktop/dataset/a.txt", sep ="\t")
#e_train = pd.read_csv("C:/Users/48784/Desktop/dataset/e.txt", sep ="\t")
#i_train = pd.read_csv("C:/Users/48784/Desktop/dataset/i.txt", sep ="\t")

#ka_test = pd.read_csv("C:/Users/48784/Desktop/dataset/test/ka.txt", sep ="\t")
#re_test = pd.read_csv("C:/Users/48784/Desktop/dataset/test/re.txt", sep ="\t")

#a_train = tf.keras.utils.normalize(a_train, axis=1)
#e_train = tf.keras.utils.normalize(e_train, axis=1)
#i_train = tf.keras.utils.normalize(i_train, axis=1)
#ka_test = tf.keras.utils.normalize(ka_test, axis=1)
#re_test = tf.keras.utils.normalize(re_test, axis=1)

#model = tf.keras.models.Sequential()
#model.add(tf.keras.layers.Flatten())
#model.add(tf.keras.layers.dense(4, activation=tf.nn.relu))
#model.add(tf.keras.layers.dense(4, activation=tf.nn.relu))
#model.add(tf.keras.layers.dense(3, activation=tf.nn.softmax))

#model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#model.fit

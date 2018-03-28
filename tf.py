import keras as k
import numpy as np
from keras.layers import Dense, Activation, Conv1D, MaxPooling1D, Flatten
from keras.models import Sequential
import data

# Importing data
train_input, train_target = \
    data.load("./data", train=True, one_khz=True, nhwc=True)
test_input, test_target = \
    data.load("./data", train=False, one_khz=True, nhwc=True)

nb_electrodes = train_input.shape[2]
nb_datapoints = train_input.shape[1]

# Creating Sequential Model
model = Sequential()

model.add(Conv1D(kernel_size=3,
                 input_shape=(nb_datapoints, nb_electrodes),
                 filters=50,
                 activation='relu'))
model.add(Conv1D(kernel_size=3, filters=50, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(kernel_size=3, filters=25, activation='relu'))
model.add(Conv1D(kernel_size=3, filters=25, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(kernel_size=3, filters=12, activation='relu'))
model.add(Conv1D(kernel_size=3, filters=12, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=50, activation='relu'))
model.add(Dense(units=2, activation='softmax'))

model.summary()

# Training model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x=train_input,
          y=np.expand_dims(train_target, -1),
          validation_split=0.2,
          batch_size=train_input.shape[0],
          epochs=200)

# Testing model
acc =\
    model.evaluate(x=test_input, y=test_target, batch_size=test_input.shape[0])
print("Testing accuracy = {0:2.1f}%".format(acc[1]*100))

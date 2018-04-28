import numpy as np
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from keras.models import Sequential, load_model
import data


model_root = "./model/"
model_name = "tf"


# Importing data
train_input, train_target = \
    data.load("./data", train=True, one_khz=True, nhwc=True, normalize=True)
test_input, test_target = \
    data.load("./data", train=False, one_khz=True, nhwc=True, normalize=True)

nb_electrodes = train_input.shape[2]
nb_datapoints = train_input.shape[1]

# Creating Sequential Model
def create_model():
    model = Sequential()
    model.add(MaxPooling1D(pool_size=5, input_shape=(nb_datapoints, nb_electrodes)))
    model.add(Conv1D(kernel_size=21,
                     filters=5,
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Flatten())
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=2, activation='softmax'))
    model.summary()
    return model
#Best so far: pool 5, conv 21 5, pool 5, dense 2
    #model.add(Dense(units=10, activation='relu'))
    #model.add(Dropout(rate=0.5))
    
model = create_model()
for i in range (0, 10):
# Training model
    model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    model.fit(x=train_input,
          y=np.expand_dims(train_target, -1),
          # validation_split=0.2,
          batch_size=20,
          epochs=50, verbose = 0)
    print("New try")
    acc_train =\
    model.evaluate(x=train_input, y=train_target, batch_size=train_input.shape[0])
    print("Training accuracy = {0:2.1f}%".format(acc_train[1]*100))
    acc_test =\
    model.evaluate(x=test_input, y=test_target, batch_size=test_input.shape[0])
    print("Testing accuracy = {0:2.1f}%".format(acc_test[1]*100))

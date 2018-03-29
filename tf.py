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
    
    model.add(Conv1D(kernel_size=3,
                     input_shape=(nb_datapoints, nb_electrodes),
                     filters=32,
                     activation='relu'))
    #model.add(Dropout(rate=0.5, noise_shape=(None,1,None)))
    model.add(Conv1D(kernel_size=3, filters=32, activation='relu'))
    #model.add(Dropout(rate=0.5, noise_shape=(None,1,None)))
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Conv1D(kernel_size=3, filters=64, activation='relu'))
    #model.add(Dropout(rate=0.5, noise_shape=(None,1,None)))
    model.add(Conv1D(kernel_size=3, filters=64, activation='relu'))
    #model.add(Dropout(rate=0.5, noise_shape=(None,1,None)))
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Conv1D(kernel_size=3, filters=128, activation='relu'))
    #model.add(Dropout(rate=0.5, noise_shape=(None,1,None)))
    model.add(Conv1D(kernel_size=3, filters=128, activation='relu'))
    #model.add(Dropout(rate=0.5, noise_shape=(None,1,None)))
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Conv1D(kernel_size=3, filters=256, activation='relu'))
    #model.add(Dropout(rate=0.5, noise_shape=(None,1,None)))
    model.add(Conv1D(kernel_size=3, filters=256, activation='relu'))
    #model.add(Dropout(rate=0.5, noise_shape=(None,1,None)))
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Flatten())
    model.add(Dense(units=100, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=100, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=2, activation='softmax'))
    
    #model.summary()
    
    return model


# Load model if existing
try:
    model = load_model(model_root + model_name, compile=False)
except OSError:
    print("Saved model not found, creating new model.")
    model = create_model()

# Training model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x=train_input,
          y=np.expand_dims(train_target, -1),
          validation_split=0.2,
          batch_size=32,
          epochs=100)

# Save model after training
print("Saving current model.")
model.save(model_root + model_name)

# Testing model
acc =\
    model.evaluate(x=test_input, y=test_target, batch_size=test_input.shape[0])
print("Testing accuracy = {0:2.1f}%".format(acc[1]*100))

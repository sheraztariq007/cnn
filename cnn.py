import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy as np
import pickle

pickle_in = open("LoadingImageDataTF/X.pickle","rb")
X = pickle.load(pickle_in)

X= X/255

pickle_in = open("LoadingImageDataTF/Y.pickle","rb")
Y = pickle.load(pickle_in)

model =  Sequential()

model.add(Conv2D(64,(3,3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# normalizing data layer

model.add(Flatten())  # now we need to flatten the data as Dense only works with 1D data
model.add(Dense(64))

# output layer

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])
X = np.array(X)
Y = np.array(Y)
model.fit(X,Y,batch_size=50,validation_split=0.1)
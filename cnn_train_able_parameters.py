from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


model = Sequential()

input_shape = (32, 32, 1)


# here first 
# 
# Conv2D(
# feature_map or output size, 
# kernel_size=(5, 5),
# activation='relu',
# input_shape=input_shape
# )

model.add(Conv2D(6, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))


model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(10, kernel_size=(3, 3),
                 activation='relu'))

model.add(Conv2D(2, kernel_size=(1, 1),
                 activation='relu'))

model.summary()

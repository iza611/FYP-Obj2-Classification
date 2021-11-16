import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
# from tensorflow.keras.utils import to_categorical
from numpy import asarray
from numpy import argmax
from matplotlib import pyplot
# from random import seed
from numpy.random import seed
from json import dump
# from tensorflow.random import set_seed
import tensorflow as tf

seed(7)
tf.random.set_seed(7)

input = np.load("all_images2.npy")
output = np.load("output2.npy")

class_names = ["Bird", "Eastern Gray Squirrel", "Eastern Chipmunk", "Woodchuck", "Wild Turkey", 
    "White_Tailed_Deer", "Virginia Opossum", "Eastern Cottontail", "Vehicle", "Striped Skunk", 
    "Red Fox", "Eastern Fox Squirrel", "Northern Raccoon", "Grey Fox", "Horse", "Dog", 
    "American Crow", "Chicken", "Domestic Cat", "Coyote", "Bobcat", "American Black Bear"]

input_train, input_test, output_train, output_test = train_test_split(input, output, test_size=0.2)
print(input_train.shape, input_test.shape, output_train.shape, output_test.shape)

n_input = input_train.shape[1:]
print(n_input)

n_output = 23

input_train = input_train.astype('float32') / 255.0
input_test = input_test.astype('float32') / 255.0

# (use this if loss='categorical_crossentropy')
# output_train = to_categorical(output_train, 23)
# output_test = to_categorical(output_test, 23)

# (set random seed instead)
# np.save("input_train.npy", input_train)
# np.save("input_test.npy", input_test)
# np.save("output_train.npy", output_train)
# np.save("output_test.npy", output_test)

model = Sequential()
model.add(Conv2D(64, (4,4), activation='relu', kernel_initializer='he_uniform', input_shape=n_input))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(32, (4,4), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(16, (3,3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(50, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(n_output, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

callback_history = model.fit(input_train, output_train, epochs=50, batch_size=1500, validation_split=0.2)

loss, acc = model.evaluate(input_test, output_test)
print('Accuracy: %.3f' % acc)

model.save('model5')
dump(callback_history.history, open("callback_history5", "w"))

image = input_train[3]
predict = model.predict(asarray([image]))
print('Predicted: class=%s' % class_names[int(argmax(predict))])

pyplot.imshow(input_train[3])
pyplot.show()

# first try:
# Accuracy: 0.169

# second try (saved as model):
# more layers, epochs: 15, batch_size: 1000
# loss: 1.4124 - accuracy: 0.6179

# third try (saved as model3):
# epochs: 50, batch_size: 1500
# loss: 472.2814 - accuracy: 0.5494

# forth try.. I need validation loss and sparse_categorical_crossentropy
# epochs: 5, batch_size: 800, validation_split=0.2

# fifth (model5)
# epochs: 50, batch_size: 1500, validation_split=0.2
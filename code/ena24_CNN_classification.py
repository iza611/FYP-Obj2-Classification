import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from numpy.random import seed
from json import dump
import tensorflow as tf

seed(7)
tf.random.set_seed(7)

input = np.load("../all_images2_normalised.npy")
output = np.load("../output2_normalised.npy")

class_names = ["Bird", "Eastern Gray Squirrel", "Eastern Chipmunk", "Woodchuck", "Wild Turkey", 
    "White_Tailed_Deer", "Virginia Opossum", "Eastern Cottontail", "null", "Vehicle", "Striped Skunk", 
    "Red Fox", "Eastern Fox Squirrel", "Northern Raccoon", "Grey Fox", "Horse", "Dog", 
    "American Crow", "Chicken", "Domestic Cat", "Coyote", "Bobcat", "American Black Bear"]

input_train, input_test, output_train, output_test = train_test_split(input, output, test_size=0.2)
print(input_train.shape, input_test.shape, output_train.shape, output_test.shape)

input_shape = input_train.shape[1:]
print(input_shape)

n_output = 23

input_train = input_train.astype('float32') / 255.0
input_test = input_test.astype('float32') / 255.0

model = Sequential()
feature_extractor = tf.keras.applications.resnet50.ResNet50(include_top = False, input_shape=input_shape)
for layer in feature_extractor.layers[:143]:
    layer.trainable = False
model.add(feature_extractor)
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(50, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(n_output, activation='softmax'))

opt = Adam(learning_rate=0.000005)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

callback_history = model.fit(input_train, output_train, epochs=100, batch_size=1500, validation_split=0.2)

loss, acc = model.evaluate(input_test, output_test)
print('Accuracy: %.3f' % acc)

model.save('model9')
dump(callback_history.history, open("callback_history9", "w"))
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from numpy import asarray
from numpy import argmax
from matplotlib import pyplot
from numpy.random import seed
from json import dump
import tensorflow as tf

seed(7)
tf.random.set_seed(7)

input = np.load("../data/input.npy")
output = np.load("../data/output.npy")

class_names = ["Bird", "Eastern Gray Squirrel", "Eastern Chipmunk", "Woodchuck", "Wild Turkey", 
    "White_Tailed_Deer", "Virginia Opossum", "Eastern Cottontail", "null", "Vehicle", "Striped Skunk", 
    "Red Fox", "Eastern Fox Squirrel", "Northern Raccoon", "Grey Fox", "Horse", "Dog", 
    "American Crow", "Chicken", "Domestic Cat", "Coyote", "Bobcat", "American Black Bear"]

input_train, input_test, output_train, output_test = train_test_split(input, output, test_size=0.2)
print(input_train.shape, input_test.shape, output_train.shape, output_test.shape)

input_shape = input_train.shape[1:]
print(input_shape)

n_output = 23

model = Sequential()
feature_extractor = tf.keras.applications.resnet_v2.ResNet50V2(include_top = False, input_shape=input_shape)
for layer in feature_extractor.layers[:86]:
    layer.trainable = False
model.add(feature_extractor)
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(50, activation='relu', kernel_initializer='he_uniform'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(n_output, activation='softmax'))

# https://www.tutorialsteacher.com/python/enumerate-method
# for i,layer in enumerate(feature_extractor.layers):
#     print(i,layer.name,layer.trainable)

# for i,layer in enumerate(model.layers):
#     print(i,layer.name,layer.trainable)

# feature_extractor.summary()
# model.summary()
# plot_model(feature_extractor, "ResNet50v2.png", show_shapes=True)

opt = Adam(learning_rate=0.000005)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

callback_history = model.fit(input_train, 
                             output_train, 
                             epochs=1000, 
                             batch_size=1500, 
                             validation_split=0.2,
                             callbacks=[
                                 EarlyStopping(monitor='val_loss',
                                               min_delta=0.0001,
                                               patience=2,
                                               restore_best_weights=True)
                                 ]
                             )

loss, acc = model.evaluate(input_test, output_test)
print('Accuracy: %.3f' % acc)

model.save('../results/model')
dump(callback_history.history, open("../results/callback_history", "w"))
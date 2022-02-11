import keras
import numpy as np
from matplotlib import pyplot as plt
from json import load
from numpy.random import seed
from sklearn.model_selection import train_test_split
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

seed(7)
tf.random.set_seed(7)

input = np.load("../all_images2_normalised.npy")
output = np.load("../output2_normalised.npy")
model = keras.models.load_model('model9')

class_names = ["Bird", "Eastern Gray Squirrel", "Eastern Chipmunk", "Woodchuck", "Wild Turkey", 
    "White_Tailed_Deer", "Virginia Opossum", "Eastern Cottontail", "Vehicle", "Striped Skunk", 
    "Red Fox", "Eastern Fox Squirrel", "Northern Raccoon", "Grey Fox", "Horse", "Dog", 
    "American Crow", "Chicken", "Domestic Cat", "Coyote", "Bobcat", "American Black Bear"]

input_train, input_test, output_train, output_test = train_test_split(input, output, test_size=0.2)

number_of_species = len(class_names) # 22
species_categories = np.arange(number_of_species) # [0,1,2,...,21]
all_images_count = output.shape[0] # 9771


### accuracy ###

loss, acc = model.evaluate(input_test, output_test)
print('Accuracy: %.3f' % acc)



### training vs validation ###

# loss & accuracy history from model training
callback_history = load(open("callback_history9", "r"))
print(callback_history["loss"][0])

# training vs validation loss
epochs = np.arange(100)
epochs = epochs + 1
loss = np.asarray(callback_history["loss"])
val_loss = np.asarray(callback_history["val_loss"])
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training loss in each epoch")
plt.legend()
plt.show()

# training vs validation accuracy
accuracy = np.asarray(callback_history["accuracy"])
val_accuracy = np.asarray(callback_history["val_accuracy"])
plt.plot(epochs, accuracy, "bo", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Accuracy in each epoch")
plt.legend()
plt.show()



### confusion matrix ###

# prep. all occurences count
all_occurences_counted = []
for species_category_id in range(number_of_species + 1): #there was also category_id = 8, it is not defined in Metadata and there is no single output with this value (so it's probably human)
    occurences = np.count_nonzero(output == species_category_id)
    all_occurences_counted.append(occurences)

removed_empty_category = all_occurences_counted.pop(8) #remove this empty category_id = 8
print(removed_empty_category)

if sum(all_occurences_counted) == all_images_count:
    print("all images counted")
else:
    print("not all images counted")

all_occurences_counted = np.asarray(all_occurences_counted)
print(all_occurences_counted)

plt.bar(species_categories, all_occurences_counted)
plt.title("Number of images per species")
plt.show()

# [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.] -> 2
def reshape_array_probability_predictions_to_int_class_prediction(array_predictions, number_of_predictions):
    predictions_reshaped = []
    for pred in range(number_of_predictions):
        predictions_reshaped.append(np.argmax(array_predictions[pred]))
    predictions_reshaped = np.asarray(predictions_reshaped)
    return predictions_reshaped

def make_predictions_in_correct_format():
    array_predictions = model.predict(input_test)
    number_of_predictions = array_predictions.shape[0]
    predictions_reshaped = reshape_array_probability_predictions_to_int_class_prediction(array_predictions, number_of_predictions)
    return predictions_reshaped

output_predictions_reshaped = make_predictions_in_correct_format()

# confusion matrix
confusion_matrix = confusion_matrix(output_test, output_predictions_reshaped)
confusion_matrix_normalised = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

plt.figure()
heatmap = sns.heatmap(confusion_matrix_normalised, annot=True, fmt='.2f', xticklabels=class_names, yticklabels=class_names)
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha="right")
heatmap.xaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=45, ha="right")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# calculate TP, FP, FN and TN
confusion_matrix = np.asarray(confusion_matrix)
columns_sum = np.sum(confusion_matrix, axis=0)
rows_sum = np.sum(confusion_matrix, axis=1)
matrix_sum = np.sum(confusion_matrix)

# # for class_name = "Bird" (0)
# TP = confusion_matrix[0,0]
# FP = columns_sum[0] - TP
# FN = rows_sum[0] - TP
# TN = matrix_sum - (TP + FN + FP)

TP = np.empty((number_of_species,))
FP = np.empty((number_of_species,))
FN = np.empty((number_of_species,))
TN = np.empty((number_of_species,))

for cat_id in range(number_of_species):
    TP[cat_id] = confusion_matrix[cat_id, cat_id]
    FP[cat_id] = columns_sum[cat_id] - TP[cat_id]
    FN[cat_id] = rows_sum[cat_id] - TP[cat_id]
    TN[cat_id] = matrix_sum - (TP[cat_id] + FP[cat_id] + FN[cat_id])

# sensitivity - you want to identify birds on images with birds
sensitivity = np.empty((number_of_species,))

for cat_id in range(number_of_species):
    sensitivity[cat_id] = TP[cat_id] / all_occurences_counted[cat_id]

plt.bar(species_categories, sensitivity)
plt.title("Sensitivity per animal category")
plt.show()

average_sensitivity = np.sum(sensitivity) / number_of_species
print("average_sensitivity")
print(average_sensitivity)

# specificity - you don't want to identify bird on an image without a bird
specificity = np.empty((number_of_species,))

for cat_id in range(number_of_species):
    specificity[cat_id] = TN[cat_id] / (FP[cat_id] + TN[cat_id])

plt.bar(species_categories, specificity)
plt.title("Specificity per animal category")
plt.show()

average_specificity = np.sum(specificity) / number_of_species
print("average_specificity")
print(average_specificity)

# precision - how much can I trust the prediction? how many good predictions?
precision = np.empty((number_of_species,))

for cat_id in range(number_of_species):
    if(TP[cat_id] + FP[cat_id] == 0):
        precision[cat_id] = 0
    else:
        precision[cat_id] = TP[cat_id] / (TP[cat_id] + FP[cat_id])

plt.bar(species_categories, precision)
plt.title("Precision per animal category")
plt.show()

average_precision = np.sum(precision) / number_of_species
print("average_precision")
print(average_precision)
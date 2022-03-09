from cgi import test
from numpy.random import choice
from matplotlib import pyplot
import numpy as np
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import json
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import img_to_array
from matplotlib import pyplot as plt
from tqdm import tqdm

# all images are cropped and saved in '../ena24_cropped2' directory
# named using ids from Metadata

# get list of all ids
# get and save output.npy
all_images = []
file = open('../../Metadata (non-human images only).json')
data = json.load(file)
ids = []
output = []
for a in range(len(data["annotations"])):
    ids.append(data["annotations"][a]["id"])
    output.append(data["annotations"][a]["category_id"])
print(len(ids))

ids = np.array(ids)
output = np.array(output)
# np.save("output.npy", output)
# output = np.load("output.npy")
# print(output.shape)

# go through each image using ids list
# process each image and add to all_images list
# count = 0
# ids = list(ids)
# for i in tqdm(ids):
# print('Processing image %d (%s.jpg)...' % (count, i))
# count += 1

# img = load_img("../../ena24_cropped2/%s.jpg" % i, target_size=(128, 128))
# x = img_to_array(img)
# x = preprocess_input(x)
# all_images.append(x)

# convert list to numpy and save
# all_images_np_array = np.array(all_images)
# print(all_images_np_array.shape)
# np.save("input.npy", all_images_np_array)

# test
# loaded_all_images_np_array = np.load("input.npy")
# print(loaded_all_images_np_array.shape) # (9772, 128, 128, 3)

# pyplot.imshow(loaded_all_images_np_array[0])
# pyplot.show()

# input = np.load("input.npy")


def count_occurances(Y):
    # count imgs
    all_occurences_counted = []
    # there was also category_id = 8, it is not defined in Metadata and there is no single output with this value (so it's probably human)
    for species_category_id in range(23):
        occurences = np.count_nonzero(Y == species_category_id)
        all_occurences_counted.append(occurences)

    # removed_empty_category = all_occurences_counted.pop(8) #remove this empty category_id = 8
    # print(removed_empty_category)

    all_images_count = Y.shape[0]
    if sum(all_occurences_counted) == all_images_count:
        print("all images counted")
    else:
        print("not all images counted")

    all_occurences_counted = np.asarray(all_occurences_counted)
    print(all_occurences_counted)
    # all_occurences_counted = np.sort(all_occurences_counted)

    # plt.bar(range(23), all_occurences_counted)
    # plt.title("Number of images per species")
    # plt.show()

    return all_occurences_counted


def get_animal_indexes(animal_label):
    animal_indexes = np.where(output == animal_label)
    animal_indexes = np.array(animal_indexes)
    animal_indexes = animal_indexes[0]
    return animal_indexes


# output = np.load("output.npy")
# input = np.load("input.npy")
all_occurences_counted1 = count_occurances(output)


# delete <292
to_delete = []
for specie in range(23):
    if(all_occurences_counted1[specie] < 292):
        to_delete.append(specie)

for specie_to_delete in to_delete:
    print("deleting specie with id %d" % specie_to_delete)
    idxs = get_animal_indexes(animal_label=specie_to_delete).tolist()
    output = np.delete(output, idxs)
    # input = remove_images(input, idxs)
    ids = np.delete(ids, idxs)


all_occurences_counted2 = count_occurances(output)

# leave 292 occurences per specie
to_reduce = []
for specie in range(23):
    if(all_occurences_counted2[specie] > 292 and all_occurences_counted2[specie] != 0):
        to_reduce.append(specie)
print(to_reduce)

for specie_to_reduce in to_reduce:
    no_imgs = all_occurences_counted2[specie_to_reduce]
    no_imgs_to_rm = no_imgs - 292
    print("reducing no images of specie with id %d from %d to 292" %
          (specie_to_reduce, no_imgs))
    idx_all = get_animal_indexes(animal_label=specie_to_reduce)
    idx_to_rm = choice(idx_all, no_imgs_to_rm, replace=False)
    output = np.delete(output, idx_to_rm)
    # input = remove_images(input, idx_to_rm)
    ids = np.delete(ids, idxs)

all_occurences_counted3 = count_occurances(output)

# Train / val / test: 187 / 47 / 58
# idxx = np.arange(292)
# np.random.shuffle(idxx)
# train_idx = idxx[0:187]
# val_idx = idxx[187:234]
# test_idx = idxx[234:292]

species_left = to_reduce
train_idx_all = []
val_idx_all = []
test_idx_all = []

for specie in species_left:
    print("Finding train & val & test idx of specie %d" % specie)

    specie_idx = np.where(output == specie)

    idxx = np.arange(292)
    np.random.shuffle(idxx)

    train_idx = idxx[0:187]
    specie_train_idx = np.take(specie_idx, train_idx)
    train_idx_all.append(specie_train_idx.tolist())

    val_idx = idxx[187:234]
    specie_val_idx = np.take(specie_idx, val_idx)
    val_idx_all.append(specie_val_idx.tolist())

    test_idx = idxx[234:292]
    specie_test_idx = np.take(specie_idx, test_idx)
    test_idx_all.append(specie_test_idx.tolist())

train_idx_all = np.array(train_idx_all).flatten()
val_idx_all = np.array(val_idx_all).flatten()
test_idx_all = np.array(test_idx_all).flatten()

# train
output_train = np.take(output, train_idx_all)
ids_train = np.take(ids, train_idx_all)

count = 0
ids_train = ids_train.tolist()
input_train = []
for i in tqdm(ids_train):
    # print('Processing image %d (%s.jpg)...' % (count, i))
    count += 1

    img = load_img("../../ena24_cropped2/%s.jpg" % i, target_size=(128, 128))
    x = img_to_array(img)
    x = preprocess_input(x)
    input_train.append(x)

input_train = np.array(input_train)
print(input_train.shape)
np.save("output_train.npy", output_train)
np.save("input_train.npy", input_train)


# val
output_val = np.take(output, val_idx_all)
ids_val = np.take(ids, val_idx_all)

count = 0
ids_val = ids_val.tolist()
input_val = []
for i in tqdm(ids_val):
    # print('Processing image %d (%s.jpg)...' % (count, i))
    count += 1

    img = load_img("../../ena24_cropped2/%s.jpg" % i, target_size=(128, 128))
    x = img_to_array(img)
    x = preprocess_input(x)
    input_val.append(x)

input_val = np.array(input_val)
print(input_val.shape)
np.save("output_val.npy", output_val)
np.save("input_val.npy", input_val)


# test
output_test = np.take(output, test_idx_all)
ids_test = np.take(ids, test_idx_all)

count = 0
ids_test = ids_test.tolist()
input_test = []
for i in tqdm(ids_test):
    # print('Processing image %d (%s.jpg)...' % (count, i))
    count += 1

    img = load_img("../../ena24_cropped2/%s.jpg" % i, target_size=(128, 128))
    x = img_to_array(img)
    x = preprocess_input(x)
    input_test.append(x)

input_test = np.array(input_test)
print(input_test.shape)
np.save("output_test.npy", output_test)
np.save("input_test.npy", input_test)

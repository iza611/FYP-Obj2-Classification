from matplotlib import pyplot
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
import json
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import img_to_array

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

output = np.array(output)
np.save("output.npy", output)
print(np.load("output.npy").shape)

# go through each image using ids list
# process each image and add to all_images list
count = 0
for i in ids:
    print('Processing image %d (%s.jpg)...' % (count, i))
    count +=1
    
    img = load_img("../ena24_cropped2/%s.jpg" % i, target_size=(128, 128))
    x = img_to_array(img)
    x = preprocess_input(x)
    all_images.append(x)

# convert list to numpy and save
all_images_np_array = np.array(all_images)
print(all_images_np_array.shape)
np.save("input.npy", all_images_np_array)

# test
loaded_all_images_np_array = np.load("input.npy")
print(loaded_all_images_np_array.shape) # (9772, 128, 128, 3)

pyplot.imshow(loaded_all_images_np_array[0])
pyplot.show()
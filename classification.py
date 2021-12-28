# replace thier data loading...
#
#fashion_mnist = tf.keras.datasets.fashion_mnist
#
#(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv
import numpy as np
import tensorflow as tf


train_dir = './images/train/'
test_dir = './images/test/'

print('##############################################')
print('TRAINING DATA...')
# OK, lets go and try and load in the training images from the correct directory...
train_img_names = glob.glob(train_dir + '*.jpg')
print('Number of training images:', len(train_img_names), '... and one Example Image:', train_img_names[0])

raw_img = Image.open(train_img_names[0])
img_data = np.asarray(raw_img)
print('Showing one example image...')
print('img_data.shape', img_data.shape)
print('img_data.dtype', img_data.dtype)
#plt.imshow(img_data)
#plt.show()

NEW_IMG_WIDTH = 350
NEW_IMG_HEIGHT = 350
# and lets try cropping...
left = (1024 - NEW_IMG_WIDTH)/2
top = (1024 - NEW_IMG_HEIGHT)/2
right = (1024 + NEW_IMG_WIDTH)/2
bottom = (1024 + NEW_IMG_HEIGHT)/2

# Crop the center of the image
raw_img = raw_img.crop((left, top, right, bottom))
img_data = np.asarray(raw_img)
print('Showing a cropped example image')
print('img_data.shape', img_data.shape)
print('img_data.dtype', img_data.dtype)
#plt.imshow(img_data)
#plt.show()


# now the target is to get our set of training images into a numpy array of shape [batch, height, width]
# and the matching labels into an array of shape [batch, 8]  8 being the coordinates for the two bounding boxes
label_dict = {}
train_label_files = glob.glob(train_dir + '*.csv')
with open(train_label_files[0], 'rt') as infile:
    reader = csv.reader(infile)
    # now process each row
    for each_row in reader:
        # use the file name as the dictionary key
        file_name = each_row[0]  # first element
        
        if file_name == 'filename':
            # we want to ignore the title row from the CSV
            continue
        
        if file_name not in label_dict:
            # add it
            label_dict[file_name] = []
        # now add in the data...
        label_dict[file_name].append(int(each_row[4]))
        label_dict[file_name].append(int(each_row[5]))
        label_dict[file_name].append(int(each_row[6]))
        label_dict[file_name].append(int(each_row[7]))
        
        # and if the labe_dict has 8 elements per file, then convert them to a numpy array
        if len(label_dict[file_name]) == 8:
            label_dict[file_name] = np.array(label_dict[file_name])


# and now we can go and create the in-sync image a label data
img_list = []
label_list = []

for each_img_name in train_img_names:
    # load each image
    raw_img = Image.open(each_img_name)
    raw_img = raw_img.crop((left, top, right, bottom))
    # and lets convert to grayscale... it seems that each of the RGB pixels is the same number
    # so lets just grab the 'R' value, and take that as the grayscale value
    gray_img = np.asarray(raw_img)[:,:,0]
    # and add this into the img_list
    img_list.append(gray_img)
    
    # and now we need to find the sync-ed label data from the dictionary
    label_file_name = each_img_name.split('\\')[1]  
    
    if label_file_name in label_dict:
        label_list.append(label_dict[label_file_name])
    else:
        raise Exception('did not find the related label data for image:', label_file_name)

    
train_images = np.array(img_list)
train_labels = np.array(label_list)

print('All training data collected into train_images, train labels, numpy arrays')
print('train_images.shape', train_images.shape)
print('train_images.dtype', train_images.dtype)
print('train_labels.shape', train_labels.shape)
print('train_labels.dtype', train_labels.dtype)
    
# and now for the labels... assume there is just one file

print('##############################################')
print('TEST DATA...')
test_img_names = glob.glob(test_dir + '*.jpg')
print('Numer of test images:', len(test_img_names), '... Example Image:', test_img_names[0])

raw_img = Image.open(test_img_names[0])
img_data = np.asarray(raw_img)
print('Showing one example image...')
print('img_data.shape', img_data.shape)
print('img_data.dtype', img_data.dtype)
#plt.imshow(img_data)
#plt.show()

# now the target is to get our set of test images into a numpy array of shape [batch, height, width]
# and the matching labels into an array of shape [batch, 8]  8 being the coordinates for the two bounding boxes

# firstly, lets throw all of the labels into a dictionary...
label_dict = {}
test_label_files = glob.glob(test_dir + '*.csv')
with open(test_label_files[0], 'rt') as infile:
    reader = csv.reader(infile)
    # now process each row
    for each_row in reader:
        # use the file name as the dictionary key
        file_name = each_row[0]  # first element
        
        if file_name == 'filename':
            # we want to ignore the title row from the CSV
            continue
        
        if file_name not in label_dict:
            # add it
            label_dict[file_name] = []
        label_dict[file_name].append(int(each_row[4]))
        label_dict[file_name].append(int(each_row[5]))
        label_dict[file_name].append(int(each_row[6]))
        label_dict[file_name].append(int(each_row[7]))
        
        if len(label_dict[file_name]) == 8:
            label_dict[file_name] = np.array(label_dict[file_name])


# and now we can go and create the in-sync image a label data
img_list = []
label_list = []

for each_img_name in test_img_names:
    # load each image
    raw_img = Image.open(each_img_name)
    raw_img = raw_img.crop((left, top, right, bottom))
    # and lets convert to grayscale
    # so lets just grab the 'R' value, and take that as the grayscale value
    gray_img = np.asarray(raw_img)[:,:,0]
    # and add this into the img_list
    img_list.append(gray_img)
    
    # and now we need to find the sync-ed label data from the dictionary
    label_file_name = each_img_name.split('\\')[1]  # maybe is OK..
    
    if label_file_name in label_dict:
        # good we found the label data
        label_list.append(label_dict[label_file_name])
    else:
        raise Exception('did not find the related label data for image:', label_file_name)

    
test_images = np.array(img_list)
test_labels = np.array(label_list)

print('All test data collected into test_images, test labels, numpy arrays')
print('test_images.shape', test_images.shape)
print('test_images.dtype', test_images.dtype)
print('test_labels.shape', test_labels.shape)
print('test_labels.dtype', test_labels.dtype)

train_images = train_images / 255.0

test_images = test_images / 255.0

# the inputs are now 1024 x 1024

# and we have 8 separate numbers coming out of the model for the 2 bounding boxes

#model = tf.keras.Sequential([
#    tf.keras.layers.Dense(64, activation='relu'),              # I get Out-Of-Memory with 128, 64
#    tf.keras.layers.Dense(64, activation='relu'),
#    tf.keras.layers.Dense(64, activation='relu'),
#    tf.keras.layers.Dense(64, activation='relu'),
#    tf.keras.layers.Dense(8, activation='relu')
#])

# Conv2D layers... 350 -> 175 -> 88 -> 44 -> 22 -> 11 -> 6 -> 3
model = tf.keras.Sequential([
    tf.keras.layers.Reshape((350,350,1)),
    tf.keras.layers.Conv2D(filters=32,
                           kernel_size=(3,3),
                           strides=(2,2),
                           padding='same',
                           activation='relu'),
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.Conv2D(filters=32,
                           kernel_size=(3,3),
                           strides=(2,2),
                           padding='same',
                           activation='relu'),
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.Conv2D(filters=16,
                           kernel_size=(3,3),
                           strides=(2,2),
                           padding='same',
                           activation='relu'),
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.Conv2D(filters=16,
                           kernel_size=(3,3),
                           strides=(2,2),
                           padding='same',
                           activation='relu'),
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.Conv2D(filters=16,
                           kernel_size=(3,3),
                           strides=(2,2),
                           padding='same',
                           activation='relu'),
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.Conv2D(filters=16,
                           kernel_size=(3,3),
                           strides=(2,2),
                           padding='same',
                           activation='relu'),
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.Conv2D(filters=16,
                           kernel_size=(3,3),
                           strides=(2,2),
                           padding='same',
                           activation='relu'),
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.Conv2D(filters=8,
                           kernel_size=(3,3),
                           strides=(1,1),
                           padding='valid',
                           activation='relu'),
    tf.keras.layers.Reshape((8,))
])

base_model = tf.keras.applications.MobileNetV2(input_shape=(350,350,3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = True


##model.compile(optimizer='adam',
##              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
##              metrics=['accuracy'])
#model.compile(optimizer='adam',
#              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
#              metrics=['accuracy'])

# just going to do a simple least squares loss
model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])

#save trained model
checkpoint_path = "./self_trained_models/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

#create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model.fit(train_images, train_labels, epochs=20, callbacks=[cp_callback])

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

predictions = model.predict(test_images)

print(predictions[0])
print(predictions[1])

# printing out predictions for visualization
test_img_show = Image.open(test_img_names[0])
test_img_data = np.asarray(test_img_show)
#plt.imshow(test_img_data)
#plt.show()
# Create figure and axes
fig, axes = plt.subplots()
# Display the image
axes.imshow(test_img_show)
# Create a Rectangle patch
rect = patches.Rectangle((predictions[0][0], predictions[0][1]), predictions[0][2]-predictions[0][0], 
        predictions[0][3]-predictions[0][1], linewidth=1, edgecolor='r', facecolor='none')
rect2 = patches.Rectangle((predictions[0][4], predictions[0][5]), predictions[0][6]-predictions[0][4], 
        predictions[0][7]-predictions[0][5], linewidth=1, edgecolor='b', facecolor='none')
# Add the patch to the Axes
axes.add_patch(rect)
axes.add_patch(rect2)
plt.show()

test_img_show = Image.open(test_img_names[1])
test_img_data = np.asarray(test_img_show)
#plt.imshow(test_img_data)
#plt.show()
# Create figure and axes
fig, axes = plt.subplots()
# Display the image
axes.imshow(test_img_show)
# Create a Rectangle patch
rect = patches.Rectangle((predictions[1][0], predictions[1][1]), predictions[1][2]-predictions[1][0], 
        predictions[1][3]-predictions[1][1], linewidth=1, edgecolor='r', facecolor='none')
rect2 = patches.Rectangle((predictions[1][4], predictions[1][5]), predictions[1][6]-predictions[1][4], 
        predictions[1][7]-predictions[1][5], linewidth=1, edgecolor='b', facecolor='none')
# Add the patch to the Axes
axes.add_patch(rect)
axes.add_patch(rect2)
plt.show()

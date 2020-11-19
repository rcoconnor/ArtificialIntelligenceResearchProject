import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import datasets, layers, models 

import numpy as np  
import matplotlib.pyplot as plt 
import pandas as pd 
import pickle

def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()



def unpickle(filename): 
    with open(filename, "rb") as fo: 
        dict = pickle.load(fo, encoding='bytes')
    return dict

def plot_images(images): 
    plt.figure(figsize=(10,10))
    for i in range(25): 
        plt.subplot(5, 5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        print(dir(images))
        print("type: " + str(type(images)))
        plt.imshow(images[i].data, cmap=plt.cm.binary)


def pre_process_images(filenames):
    num_samples = 50000
    # load the data
    #images = np.array((num_samples, 3072), dtype='uint8')
    #labels = np.array((num_samples,), dtype='uint8')
    images = np.array([])
    labels = np.array([])
    test_images = []
    test_labels = []
    print("origshape: " + str(np.shape(images)))
    for each_file in filenames:
        file_dict = unpickle(each_file)
        if each_file == "cifar-10-batches-py/test_batch": 
            test_images = file_dict[b'data'] 
            test_labels = file_dict[b'labels'] 
        else:
            images = np.append(images, file_dict[b'data'])
            labels = np.append(labels, file_dict[b'labels'])
            print("labels shape: " + str(np.shape(labels)))
    images = images.reshape((50000, 3072))
    print("test: " + str(np.shape(test_images)))
    print("labe: " + str(np.shape(test_labels)))
    return (images, labels), (test_images, test_labels)

filenames = ["cifar-10-batches-py/data_batch_1",
             "cifar-10-batches-py/data_batch_2",
             "cifar-10-batches-py/data_batch_3",
             "cifar-10-batches-py/data_batch_4",
             "cifar-10-batches-py/data_batch_5",
             "cifar-10-batches-py/test_batch"]

print("creating model....", end='')
model = keras.models.Sequential()
model.add(layers.Input(shape=(3072,)))
model.add(layers.Dense(3072, activation="relu"))
model.add(layers.Dense(1024, activation="relu"))
model.add(layers.Dense(32, activation="relu"))
#model.add(layers.Dense(84, activation="relu"))
#model.add(layers.Dense(192, activation="relu"))
#model.add(layers.Dense(96, activation="relu"))
#model.add(layers.Dense(48, activation="relu"))
#model.add(layers.Dense(24, activation="relu"))
model.add(layers.Dense(12, activation="relu"))
model.add(layers.Dense(10))
model.add(layers.Activation("softmax"))
print("created!")

(cifar_images, cifar_labels), (test_images, test_labels) = pre_process_images(filenames)

# convert the integers to floating point numbers 
cifar_images = cifar_images.astype('float32')
test_images = test_images.astype('float32')
cifar_images /= 255
test_images /= 255

print()
print("train_images shape: " + str(np.shape(cifar_images)))
print("train_labels shape: " + str(np.shape(cifar_labels)))
print("test_images shape: " + str(np.shape(test_images)))
print("test_labels shape: " + str(np.shape(test_labels)))
print()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

print("training model....", end='')

#history = model.fit(cifar_images, cifar_labels, epochs=10, validation_data=(test_images, test_labels))
history = model.fit(cifar_images, cifar_labels, epochs=2)

"""
plt.plot(history.history['accuracy'], label='accuracy')
#plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
"""

print("image shape: " + str(np.shape(test_images)))
print("label shape: " + str(np.shape(test_labels)))
print("type: " + str(type(test_image)))

print("cifar shape: " + str(cifar_images))

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print("loss: " + test_loss)
print("accuracy: " + test_acc)

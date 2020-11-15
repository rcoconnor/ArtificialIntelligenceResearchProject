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


def pre_process_images():
    # load the data
    images_dict = unpickle("cifar-10-batches-py/data_batch_1")
    image_data = images_dict[b'data']
    image_filenames = images_dict[b'filenames']
    image_labels = images_dict[b'labels']
    #reshape the images 
    image_data = image_data.reshape((len(images_dict[b'data']), 3, 32, 32))
    num_plot = 5
    index = 0
    f, ax = plt.subplots(num_plot, num_plot)
    for m in range(num_plot): 
        for n in range(num_plot): 
            ax[m, n].imshow(image_data[index])
            ax[m, n].get_xaxis().set_visible(False)
            ax[m, n].get_yaxis().set_visible(False)

    f.subplots_adjust(hspace=0.1)
    f.subplots_adjust(wspace=0)
    plt.show()

pre_process_images()
#plot_images(image_data)

"""
for index in range(10): 
    plt.subplot(5, 5, index+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    print(dir(images))
    print("types: " + str(type(images)))
    plt.imshow(images[index].data, cmap=plt.cm.binary)
"""
#plot_images(images)

exit()    

# Import the fashion Mnist dataset 
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

print("shape: " + str(np.shape(test_labels)))

# convert the integers to floating point numbers 
train_images, test_images = train_images / 255.0, test_images / 255.0

print("creating model....", end='')
model = keras.models.Sequential()
model.add(layers.Input(shape=(768,)))
model.add(layers.Dense(768, activation="relu"))
#model.add(layers.Dense(384, activation="relu", kernel_initializer="uniform",))
#model.add(layers.Dense(10))
#model.add(layers.Activation("softmax"))
print("created!")

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

print("training model....", end='')
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print("loss: " + test_loss)
print("accuracy: " + test_acc)

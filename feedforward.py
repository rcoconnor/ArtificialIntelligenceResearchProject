import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import datasets, layers, models 

import numpy as np  
import matplotlib.pyplot as plt 

def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()


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
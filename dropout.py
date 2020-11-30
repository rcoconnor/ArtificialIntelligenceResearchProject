"This program creates a convolutional Neural Network and trains it"
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def unpickle(filename):
    """this is our docstring"""
    with open(filename, "rb") as file:
        the_dict = pickle.load(file, encoding='bytes')
    return the_dict


def load_cifar_data(filename):
    """Loads the cifar data"""
    with open(filename, "rb") as the_file:
        file_dict = pickle.load(the_file, encoding='latin1')
        the_images = file_dict['data']
        the_images = the_images.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")

        the_labels = np.array(file_dict['labels'])
        return the_images, the_labels

def get_cifar_data():
    "returns a tuple containing training data and the labels"
    filenames = ["cifar-10-batches-py/data_batch_1",
             "cifar-10-batches-py/data_batch_2",
             "cifar-10-batches-py/data_batch_3",
             "cifar-10-batches-py/data_batch_4",
             "cifar-10-batches-py/data_batch_5",
             "cifar-10-batches-py/test_batch"]
    data=[]
    the_labels=[]
    for i in range(6):
        the_images, new_labels = load_cifar_data(filenames[i])
        data = np.append(data, the_images)
        the_labels = np.append(the_labels, new_labels)
        data = data.reshape(((i+1) * 10000, 32, 32, 3))
    return data, the_labels


def create_model():
    """returns the model we are creating"""
    the_model = models.Sequential()
    the_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    the_model.add(layers.MaxPooling2D(2, 2))
    the_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    the_model.add(layers.MaxPooling2D(2, 2))
    the_model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Add dense layers on top
    the_model.add(layers.Flatten())
    the_model.add(layers.Dense(64, activation='relu'))
    the_model.add(layers.Dense(10))
    return the_model


def plot_model_performance(the_history):
    "Plots the models performance"
    plt.plot(the_history.history['accuracy'], label='accuracy')
    plt.plot(the_history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel("epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

def plot_images(the_images):
    "plots the images to the screen"
    the_images = np.array(the_images, np.int32)
    fig, axes1 = plt.subplots(5, 5, figsize=(3,3))
    index = 0
    for j in range(5):
        for k in range(5):
            axes1[j][k].set_axis_off()
            axes1[j][k].imshow(the_images[index])
            index += 1
    plt.show()
    fig.show()



class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']




images, labels = get_cifar_data()
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.2)

print("shape: " + str(len(images)))

model = create_model()
print("N_trian: " + str(len(images)))
print("steps per epoch: " + str(len(images)))

#plot_images(images)
# Gradually decrease the learning rate over time




learning_rate_fn = tf.keras.optimizers.schedules.InverseTimeDecay(0.1,
        decay_steps= 1.0,
        decay_rate = 0.5)

STEPS_PER_EPOCH = len(images)



model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print("training")
history = model.fit(x_train, y_train, epochs=20,
                   validation_data=(x_test, y_test))
print("done!")

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

plot_model_performance(history)

print("Test Loss: " + str(test_loss))
print("Test Accuracy: " + str(test_acc))

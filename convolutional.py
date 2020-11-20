import tensorflow as tf 
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pickle 

def unpickle(filename):
    with open(filename, "rb") as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_cifar_data(filename):
    with open(filename, "rb") as fo: 
        file_dict = pickle.load(fo, encoding='latin1')
        images = file_dict['data']
        images = images.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")

        labels = np.array(file_dict['labels'])
        return images, labels

def get_cifar_data(): 
    filenames = ["cifar-10-batches-py/data_batch_1",
             "cifar-10-batches-py/data_batch_2",
             "cifar-10-batches-py/data_batch_3",
             "cifar-10-batches-py/data_batch_4",
             "cifar-10-batches-py/data_batch_5",
             "cifar-10-batches-py/test_batch"]
    data=[]
    labels=[]
    for i in range(6):
        images, new_labels = load_cifar_data(filenames[i])
        data = np.append(data, images)
        labels = np.append(labels, new_labels)
        data = data.reshape(((i+1) * 10000, 32, 32, 3))
    return data, labels


def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Add dense layers on top 
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    return model


def plot_model_performance(history): 
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel("epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

def plot_images(images):
    images = np.array(images, np.int32)
    fig, axes1 = plt.subplots(5, 5, figsize=(3,3))
    index = 0
    for j in range(5): 
        for k in range(5): 
            axes1[j][k].set_axis_off()
            axes1[j][k].imshow(images[index])
            index += 1
    plt.show()


# FIXME: Create a separate testing and training set 

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

images, labels = get_cifar_data()
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.2)

model = create_model()

#plot_images(images)
print("compiling.....", end='')
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
print("done!")

print("training")
history = model.fit(x_train, y_train, epochs=20, 
                   validation_data=(x_test, y_test))
print("done!")

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

plot_model_performance(history)

print("Test Loss: " + str(test_loss))
print("Test Accuracy: " + str(test_acc))

# tensorflow & tf.keras
import tensorflow as tf
from tensorflow import keras

# helper library
import numpy as np
import matplotlib.pyplot as plt

# fashion mnist data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

'''
plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
plt.show()
'''

# Data preprocessing [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # 28 * 28 = 784
    keras.layers.Dense(1024, activation='sigmoid'),  # densely-connected, sigmoid
    keras.layers.Dense(512, activation='sigmoid'),  # densely-connected, sigmoid
    keras.layers.Dense(10, activation='softmax')  # densely-connected, softmax
])

# model compile
# loss function: minimize error
# optimizer: update the model based on loss function
# metrics: to monitor training and test
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train the model
model.fit(train_images, train_labels, epochs=15)

# test accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest Accuracy: ', test_acc)

predictions = model.predict(test_images)
print(predictions[0])
print('Prediction: ', np.argmax(predictions[0]))
print('Label: ', test_labels[0])


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'  # correct
    else:
        color = 'red'   # wrong

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]), color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from utils import load_galaxy_data

import app

# Input DATA and labels
input_data, labels = load_galaxy_data()

# Data's info
print(input_data.shape, labels.shape)

# Split to train
X_train, X_test, Y_train, Y_test = train_test_split(input_data,labels, test_size = 0.2, shuffle = True, random_state = 222, stratify = labels)

# Data Generator
data_generator = ImageDataGenerator(rescale = 1./255)

# Iterators
training_iterator = data_generator.flow(X_train, Y_train, batch_size = 5)
validation_iterator = data_generator.flow(X_test, Y_test, batch_size = 5)

# Build the model
model = tf.keras.Sequential()

# Add the input layer
model.add(tf.keras.Input(shape = (128, 128, 3)))

model.add(tf.keras.layers.Conv2D(8, 3, strides = 2, activation = "relu"))

model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2), strides = 2))

model.add(tf.keras.layers.Conv2D(8, 3, strides = 2, activation = "relu"))

model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2), strides = 2))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(16, activation = "relu"))
# Add the output layer
model.add(tf.keras.layers.Dense(4, activation = "softmax"))

# Compile the model
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), loss = tf.keras.losses.CategoricalCrossentropy(), metrics = [tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.AUC()])

print(model.summary())

# Fit and train the model
model.fit(training_iterator, steps_per_epoch = len(X_train)/5, epochs = 8, validation_data = validation_iterator, validation_steps = len(X_test)/5)

from visualize import visualize_activations
visualize_activations(model, validation_iterator)

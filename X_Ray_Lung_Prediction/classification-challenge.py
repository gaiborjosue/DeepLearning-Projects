import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy

BATCH_SIZE = 32

training_data_generator = ImageDataGenerator(rescale=1./255, zoom_range=0.1, rotation_range=25, width_shift_range=0.05, height_shift_range=0.05)

# The data has 3 classes
training_iterator = training_data_generator.flow_from_directory('X_Ray_Lung_Prediction\\Covid19-dataset\\train', batch_size=BATCH_SIZE, class_mode='categorical', color_mode='grayscale')

training_iterator.next()

validation_data_generator = ImageDataGenerator()

validation_iterator = validation_data_generator.flow_from_directory('X_Ray_Lung_Prediction\\Covid19-dataset\\train', batch_size=BATCH_SIZE, class_mode='categorical', color_mode='grayscale')


model = Sequential()

model.add(layers.Input(shape=(256, 256, 1)))

model.add(layers.Conv2D(5, 5, strides=3, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(layers.Conv2D(3, 3, strides=1, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(3, activation='softmax'))

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), loss = tf.keras.losses.CategoricalCrossentropy(), metrics = [tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.AUC()])

print(model.summary())

# Fit and train the model
model.fit(training_iterator, steps_per_epoch = training_iterator.samples/BATCH_SIZE, epochs = 5, validation_data = validation_iterator, validation_steps = validation_iterator.samples/5)

# Evaluate the model
print(model.evaluate(validation_iterator))

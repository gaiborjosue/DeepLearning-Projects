from tensorflow.keras import layers
layer = layers.Dense(3) #3 is the number we chose

print(layer.weights) #we get empty weight and bias arrays because tensorflow doesn't know what the shape is of the input to this layer


input = tf.ones((1338, 11)) #13388 samples, 11 features as in our dataset
layer = layers.Dense(3) # a fully-connected layer with 3 neurons
output = layer(input) #calculate the outputs
print(layer.weights) #print the weights
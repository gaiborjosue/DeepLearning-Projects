Preparing the data for learning:
  separating features from labels using array slicing
  determining the shape of your data
  preprocessing the categorical variables using one-hot encoding
  splitting the data into training and test sets
  scaling the numerical features

Designing a Sequential model by chaining InputLayer() and the tf.keras.layers.Dense layers. InputLayer() was used as a placeholder for the input data. The output layer in this case needed one neuron since we need a prediction of a single value in the regression. And finally, hidden layers were added with the relu activation function to handle complex dependencies in the data.

Choosing an optimizer using keras.optimizers with a specific learning rate hyperparameter.

Training the model - using model.fit() to train the model on the training data and training labels.

Setting the values for the learning hyperparameters: number of epochs and batch sizes.

Evaluating the model using model.evaluate() on the test data.
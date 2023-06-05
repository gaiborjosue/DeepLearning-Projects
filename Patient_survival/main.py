import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
import numpy as np

# Import the data
data = pd.read_csv("heart_failure.csv")

# Get essential info
print(data.info())

# Counter for the death event column
print(Counter(data["death_event"]))

# Extract data, for y and x
y = data["death_event"]
x = data[['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time']]

# Convert categorical features in DF to one-hot encoding
x = pd.get_dummies(x)

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

# Init a CT object with Standard Scaler
ct = ColumnTransformer([("numeric", StandardScaler(), ['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time'])])

# Train the instance ct with the training data
X_train = ct.fit_transform(X_train)

X_test = ct.transform(X_test)

# Initialize label Encoder
le = LabelEncoder()

Y_train = le.fit_transform(Y_train.astype(str))
Y_test = le.transform(Y_test.astype(str))

# Transform the encoded training labels Y train into a binary vector and assign the result back to Y_train
Y_train  = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

# Initialize a Sequential model instance
model = Sequential()

# Add the input layer
model.add(InputLayer(input_shape=(X_train.shape[1],)))

# Create a hidden layer of instance of Dense with relu
model.add(Dense(12, activation = "relu"))

# Create the output layer
model.add(Dense(2, activation = "softmax"))

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

# Lets fit the model
model.fit(X_train, Y_train, epochs = 100, batch_size = 16, verbose = 1)

# Evaluate the model
loss, acc = model.evaluate(X_test, Y_test, verbose = 0)

# Predict
y_estimate = model.predict(X_test)

# Select the indices of the true classes for each label
y_estimate = np.argmax(y_estimate, axis = 1)
y_true = np.argmax(Y_test, axis = 1)

# Print additional models
print(classification_report(y_true, y_estimate))
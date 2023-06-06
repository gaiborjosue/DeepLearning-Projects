import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense

# Read the dataset
dataset = pd.read_csv("life_expectancy.csv")

# Drop the country column, since we only want a general model applicable for all countries
dataset = dataset.drop(['Country'], axis = 1)

# Get the labels
labels = dataset.iloc[:, -1]

# Features
features = dataset.iloc[:, 0:-1]

# One-hot encoding for categorical values in features
features = pd.get_dummies(features)

# Split the data into train and test
from sklearn.model_selection import train_test_split

features_train, labels_train, features_test, labels_test = train_test_split(features, labels, test_size = 0.3, random_state=42)

# standardize/normalize your numerical features
numerical_features = features.select_dtypes(include=['float64', 'int64'])

numerical_columns = numerical_features.columns

ct = ColumnTransformer([("only numeric", StandardScaler(), numerical_columns)], remainder='passthrough')

# Transform the test data features_test using ct
features_train_scaled = ct.fit_transform(features_train)
features_test_scaled = ct.transform(features_test)

# Create the model
my_model = Sequential()

# Input layer
input = InputLayer(input_shape = (features.shape[1], ))
my_model.add(input)

# Hidden layer
my_model.add(Dense(64, activation="relu"))

# Output layer with one neuron since we only need a single output for a regression prediction
my_model.add(Dense(1))

# Print the summary of the model
print(Sequential.summary())


###### Optimizer and compiling the model
from tensorflow.keras.optimizers import Adam
opt = Adam(learning_rate = 0.1)
model.compile(loss='mse', metrics=['mae'], optimizer = opt)

##### Fit and evaluate the model
model.fit(features_train_scaled, labels_train, epochs = 40, batch_size = 1, verbose = 1)

res_mse, res_mae = model.evaluate(features_test_scaled, labels_test, verbose = 0)

print(res_mse, res_mae)
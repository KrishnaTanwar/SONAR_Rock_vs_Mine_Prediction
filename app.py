# Importing the Dependencies

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data Collection & Preprocessing

# loading the dataset to a pandas dataframe
data = pd.read_csv('sonardata.csv',header=None)

print(data.head())

# Number of rows and columns
print(data.shape)

# Describe = Statistical measures of the data
print(data.describe())

print(data[60].value_counts())

# M--> Mine
# R--> Rock

print(data.groupby(60).mean())

print(data[60].value_counts())


# Separating data and labels

X = data.drop(columns = 60, axis = 1)
Y = data[60]

print(X)
print(Y)

# Training and Testing data

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1)
print(X.shape,X_train.shape,X_test.shape)

# Model Training --> LogisticRegression

model = LogisticRegression()

#training the LogisticRegression model with Training data
model.fit(X_train,Y_train)

# Model Evaluation

# Accuracy on training data

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)
print("Accuracy on Training data : ", training_data_accuracy)

#Accuracy on test data

X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print("Accuracy on Training data : ", testing_data_accuracy)

# Making a predictive system

input_data = (0.0090,0.0062,0.0253,0.0489,0.1197,0.1589,0.1392,0.0987,0.0955,0.1895,0.1896,0.2547,0.4073,0.2988,0.2901,0.5326,0.4022,0.1571,0.3024,0.3907,0.3542,0.4438,0.6414,0.4601,0.6009,0.8690,0.8345,0.7669,0.5081,0.4620,0.5380,0.5375,0.3844,0.3601,0.7402,0.7761,0.3858,0.0667,0.3684,0.6114,0.3510,0.2312,0.2195,0.3051,0.1937,0.1570,0.0479,0.0538,0.0146,0.0068,0.0187,0.0059,0.0095,0.0194,0.0080,0.0152,0.0158,0.0053,0.0189,0.0102)

# changing the input data to a numpy array

input_data_as_numpy_array = np.asarray(input_data)

# Reshape the np array as we are predicting for one instance

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 'R'):
    print("The object is a Rock")
else:
    print("The object is a Mine")
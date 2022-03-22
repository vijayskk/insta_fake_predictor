import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.linear_model as lm
import tensorflow as tf
from tensorflow import keras
data = pd.read_csv('test.csv')

data = data.dropna()

print(data)
X = data.drop(['fake'], axis=1)
Y = data['fake']

model = lm.LinearRegression()

Xtrain , Xtest , Ytrain , Ytest = sklearn.model_selection.train_test_split(X , Y , test_size = 0.1)
# model.fit(Xtrain, Ytrain)

# acc = model.score(Xtest, Ytest)
# print(acc)


model = keras.Sequential([
    tf.keras.layers.Dense(12, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])


model.fit(Xtrain,Ytrain,epochs=1000)



joblib.dump(model, 'model.joblib')
# model = joblib.load('model.joblib')

pred = model.predict(Xtest)
index = 0
pos = 0
neg = 0
print(Ytest)
for i in Ytest:
    j = pred[index][0]
    print(str(round(j)) + " -> " + str(i))
    index += 1
    if(round(j) == i):
        pos += 1
    else:
        neg += 1

print("Positive: " + str(pos) + " Negative: " + str(neg))
print("Accuracy: " +str(pos / (pos + neg)))


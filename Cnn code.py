#This model is a mix of ANN and CNN Neural Networks on the tensorfow cifa10 dataset. The model decreases loss function and classifies images based on the data.

#Frameworks: Tensorflow,Keras
#Libraries:Numpy,Pandas,Matplotlib,Tensorflow,sklearn,etc.'


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#importing libraries

import tensorflow as tf 
from tensorflow.keras import datasets,models,layers
import matplotlib.pyplot as plt 


Cifar10 dataset loading
(x_train,y_train),(x_test,y_test)=datasets.cifar10.load_data()

#train dataset shape
x_train.shape

y_train.shape

#test dataset shape
x_test.shape

y_test.shape


#Data Vizualisation
x_train[0]

y_train[0]

plt.imshow(x_train[0])

plt.imshow(x_train[1])

plt.imshow(x_train[2])


#Normalising the dataset

x_train=x_train/255.0
x_test=x_test/255.0


#Artificial Neural Network

ann =models.Sequential([
    layers.Flatten(input_shape=(32,32,3)),
    layers.Dense(1000,activation='relu'),
    layers.Dense(1000,activation='relu'),
    layers.Dense(10,activation='softmax')
    
])


#Compiling the model using Stochastic Gradient Decent optimizer and sparse_categorical_crossentropy


ann.compile(optimizer='SGD',
           loss ='sparse_categorical_crossentropy',
           metrics=['accuracy'])


#Model fitting with 10 epochs
ann.fit(x_train,y_train,epochs=10)


#Making a classification report using sklearn library

from sklearn.metrics import classification_report 
y_pred =ann.predict(x_test)
y_pred_classes=[np.argmax(element) for element in y_pred]
print('classification_report',classification_report(y_test,y_pred_classes))



#Convolutional Neural Network
#Building Model
#Setting parameters


cnn = models.Sequential([
    layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(32,32,3)),
    layers.MaxPooling2D(2,2),
    
    
    
    layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D(2,2),
    
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(10,activation='softmax')
    
])


#Compilation using adam optimizer
cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])


#model fitting with 5 epochs
cnn.fit(x_train,y_train,epochs=5)

#Classification report for cnn

from sklearn.metrics import classification_report 
y_pred =cnn.predict(x_test)
y_pred_classes=[np.argmax(element) for element in y_pred]
print('classification_report',classification_report(y_test,y_pred_classes))


#Evaluation
cnn.evaluate(x_test,y_test)


#Prediction
y_pred=cnn.predict(x_test)
y_pred[:5]


#This Model is fully programmed by Dev kumar


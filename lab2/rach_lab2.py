from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cross_validation import cross_val_score, train_test_split
import pandas as pd
import tensorflow as tf
from tensorflow import keras


fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')

#Function main()
def main():

   #Randomly generate x1 and x2, based on which, calculate y1 and y2
   x1 = np.random.rand(1, 100) * 100
   x2 = np.random.rand(1, 100) * 100
   y1 = ((x1 - x2)*(x1 - x2))/len(x1[0])
   y2 = np.sqrt((x1*x1 + x2*x2))

   #Write the arrays into a CSV where each column represents an array
   df = pd.DataFrame({"x1" : x1[0], "x2" : x2[0], "y1" : y1[0], "y2" : y2[0]})
   df.to_csv("mydata.csv", index=True)

   #Read data from the CSV into a Pandas dataframe
   df = pd.read_csv('mydata.csv', index_col=0)

   data1 = df[['y1', 'y2', 'x1']]		#selects all rows of the columns: y1, y2, x1
   data2 = df[['y1', 'y2', 'x2']]		#selects all rows of the columns: y1, y2, x2
   
   #Split the data into training and test data using an 80-20 split
   x_train, x_test, y_train, y_test = train_test_split(data1[['y1', 'y2']], data1['x1'], test_size=0.2)
   print(len(x_train), len(x_test), len(y_train), len(y_test))

   #Normalizing the features using y1 and y2. Test Data is NOT used when calculating the mean and std
   mean = y_train.mean(axis=0)
   std = y_train.std(axis=0)
   y_train = (y_train - mean) / std
   y_test = (y_test - mean) / std

   model = build_model(y_train)
   model.summary()


#Function to build the model
def build_model(y_train):
  
  #Sequential model using 2 densely connected layers and an o/p layer which returns 1 single, continuous value
  model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(y_train.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model

   
main()
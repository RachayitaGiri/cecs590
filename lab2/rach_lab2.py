from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cross_validation import cross_val_score, train_test_split
import pandas as pd


fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')

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

   data1 = df[['x1', 'x2', 'y1']]		#selects all rows of the columns: x1, x2, y1
   data2 = df[['x1', 'x2', 'y2']]		#selects all rows of the columns: x1, x2, y2
   
   #Split the data into training and test data using an 80-20 split
   x_train, x_test, y_train, y_test = train_test_split(data1[['x1', 'x2']], data1['y1'], test_size=0.2)
   print(len(x_train), len(x_test), len(y_train), len(y_test))
   
   model = build_model(x_train)
   model.summary()


#Function to build the model
def build_model(x_train):
  
  #Sequential model using 2 densely connected layers and an o/p layer which returns 1 single, continuous value
  model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(x_train.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model

main()
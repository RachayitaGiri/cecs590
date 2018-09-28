from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cross_validation import cross_val_score, train_test_split
import pandas as pd
import tensorflow as tf
from tensorflow import keras


# fig = plt.figure(1)
# ax = fig.add_subplot(111, projection='3d')

#Function main()
def main():

   #Randomly generate x1 and x2, based on which, calculate y1 and y2
   x1 = np.random.rand(1, 10000) * 100
   x2 = np.random.rand(1, 10000) * 100
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
   y_train, y_test, x_train, x_test = train_test_split(data1[['y1', 'y2']], data1['x1', 'x2'], test_size=0.2)
   print(len(x_train), len(x_test), len(y_train), len(y_test))
   print(x_test)
   #Normalizing the features using y1 and y2. Test Data is NOT used when calculating the mean and std
   mean = y_train.mean(axis=0)
   std = y_train.std(axis=0)
   y_train = (y_train - mean) / std
   y_test = (y_test - mean) / std

   model = build_model(y_train)
   model.summary()

   # Store training stats
   history = model.fit(y_train, x_train, epochs=EPOCHS, validation_split=0.2, verbose=0, callbacks=[PrintDot()])

   #The patience parameter is the amount of epochs to check for improvement
   early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

   #Plotting the model's training progress stored in the 'history' object no show
   plot_history(history)

   [loss, mae] = model.evaluate(y_test, x_test, verbose=0)

   print("\nTesting set Mean Abs Error:{:7.2f}".format(mae))

   test_predictions = model.predict(y_test)
   print(test_predictions)

   plt.scatter(x_test, test_predictions)
   plt.xlabel('Actual x1 values')
   plt.ylabel('Predicted x1 values')
   plt.axis('equal')
   plt.xlim(plt.xlim())
   plt.ylim(plt.ylim())
   _ = plt.plot([-100, 100], [-100, 100])

   plt.show()

#Function to build the model
def build_model(y_train):
  
  #Sequential model using 2 densely connected layers and an o/p layer which returns 1 single, continuous value
  model = keras.Sequential([
    keras.layers.Dense(256, activation=tf.nn.relu, input_shape=(y_train.shape[1],)),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dense(2)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()
  plt.ylim([0, 5])
  plt.show()


EPOCHS = 500

main()
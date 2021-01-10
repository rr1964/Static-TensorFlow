
# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import TF_Static_DataBuild as dataBuild

print(tf.__version__)

##fullData = np.load("fullData.npy")

fullData = dataBuild.get_full_data()

##Get the train/test data as well as the labels.
trainData = fullData[:20000, :-1]
testData = fullData[20000: , :-1]

trainLabel = fullData[:20000, -1]
testLabel = fullData[20000:, -1]

##Do a fast check to make sure that everything is the dimensions we think they are.
print(trainData.shape)
print(testData.shape)

print(trainLabel.shape)
print(testLabel.shape)


model = tf.keras.Sequential([
    ##tf.keras.layers.Flatten(input_shape=(40, 40)), ## Flatten means go from 40 x 40 to 1 x 1600. ##Our data is already flat, so skip this. s

    tf.keras.layers.InputLayer(input_shape=(1600,)),
    tf.keras.layers.Dropout(rate = 0.07, noise_shape=None, seed=None), ##I'm adding a dropout layer to prevent overfitting. Definitely improves accuracy on test.
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)), ## Dense means fully connected.
    tf.keras.layers.Dense(64, activation='relu'),  ## Dense means fully connected.
    tf.keras.layers.Dense(dataBuild.numClasses) ## The final layer will return a logits vector with length numClasses.
    ## Each entry contains a score that indicates the current image belongs to one of the classes.
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(trainData, trainLabel, epochs=15)

testLoss, testAcc = model.evaluate(testData,  testLabel, verbose=2)

print('\nTest accuracy:', testAcc)

## Can get around 90% accuracy on test set when the parameters are as follows:
##          numClasses = 10
##          scale_k = 6
##          scale_z = 0.2

#dataBuild.plot_sample()
#dataBuild.save_data()



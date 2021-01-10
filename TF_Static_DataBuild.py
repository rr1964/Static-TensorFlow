## Creating a data set to be used with TF_Static.py

import numpy as np
import matplotlib.pyplot as plt


##np.random.seed(100)

##First create the canonical examples of the classes.
## The images are 40 x 40 = 1600 pixels. Greyscale

numClasses = 10
numImgPix = 1600

canon = np.random.rand(numClasses, numImgPix)

## Maybe have 30,000 samples. 20k training, 10k testing.
samples = 30000
scale_k = 6
scale_z = 0.2

dataHold = []

for i in range(samples):

    ## A +/-1 and 0 vector. Can also just be changed to be +/-1
    negZeroOne = np.random.choice([-1,0,1], size = numImgPix, replace=True, p = [0.45, 0.1, 0.45])

    c_k = np.random.randint(0,numClasses,1) ## Both bounds are inclusive

    c_z = np.random.randint(0,numClasses,1) ## Both bounds are inclusive

    line = np.add(canon[c_k,],  negZeroOne * np.random.rand(numImgPix)* scale_k)

    ##This overlays a second random class image on top of the main image.
    line = np.add(line,  scale_z * canon[c_z] * np.random.choice([0,1], size = 1, p = [0.95, 0.05]))

    ## Scale the generated data to be between 0 and 1.
    line = np.interp(line, (line.min(), line.max()), (0, 1))

    line = np.append(line, [c_k], axis=1)##Append the class label.

    dataHold.append(line)

data = np.stack(dataHold, axis=1).reshape(samples, numImgPix + 1) ##Add 1 for the class label.

print(data.shape) ##Should be (samples, numImgPix+1)

print(int(data.item((1, 1600)))) ## A debug print to ensure we are getting the right classes

def get_full_data():
    return data

def plot_sample():

    for j in range(12):

        j+=1
        ##print(i)

        plt.subplot(4,6,2*j - 1)
        plt.imshow(data[j, :-1].reshape(40,40), interpolation='nearest')

        plt.subplot(4,6,2*j)
        plt.imshow(canon[int(data.item((j, 1600)))].reshape(40,40), interpolation='nearest')

    plt.show()


def save_data(as_text = False):

    if as_text:
        np.savetxt("fullData.csv", data, delimiter=',', fmt = '%.20f') ## Only use this if you need a csv. Much slower and bulkier.

    else:
        np.save("fullData.npy", data)  ##This is a numpy binary. Can only be opened with numpy. But very compact and fast to write.






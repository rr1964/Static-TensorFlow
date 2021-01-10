# Static-TensorFlow
A simple example of synthetic static data and using a neural network in tf.keras to classify images. 


I generate ten canonical images that define ten classes. 
Based off of these canonical images, I then generate 30,000 images that are random alterations of these canon images.

The images are 40 x 40 pixel images with intensity values in the interval [0,1). I treat these images as 1 x 1600 numpy arrays.

Each of the 30,000 sample "image" is generated as follows:

- Let **q**_k be a 1 x 1600 vector containing the values +/-1 and 0, randomly generated with probablities P(1) = P(-1) = 0.45 and P(0) = 0.1. 
- Let c_k and c_z be randomly selected integers between 0 and 9. Each of these represent class labels. **C_k** and **C_z** will represent the canoncial images for the associated classes.
- Let s_k and s_z be non-negative scale parameters. Let t be either 0 or 1, with P(t = 0) = 0.95. 
- Generate a sample of the class given by c_k by the following formula, where **v** is a randomly generated 1 x 1600 vector with entries all in the interval [0,1) (sampling is uniform):
  * Sample = **C_k** + s_k(**q** * **v**) + s_z(t**C_z**), where * is component wise multiplication resulting in a 1 x 1600 vector. 
- The values of the generated sample are then scaled to be between 0 and 1. 

Each sample image is some jittered version of a canonical image. In about 5% of sample images, a second canonical image is (lightly) overlaid onto the first. 

I fit a nerual network on 20,000 training images. The test set is 10,000 images. Setting s_k = 6 and s_z = 0.2 results in being able to obtain about 90% test accuracy on most synthesized data sets. 



Abstract

Artificial Intelligence has been witnessing a monumental growth in bridging the gap between the capabilities of humans and machines. One of many such areas is the domain of Computer Vision.
The advancements in Computer Vision with Deep Learning has been constructed and perfected with time, primarily over one particular algorithm — a Convolutional Neural Network (CNN).

In neural networks, Convolutional neural network (ConvNets or CNNs) is one of the main categories for image recognition, image classifications, Objects detections, recognizing faces etc., are some of the areas where CNNs are widely used.

The CIFAR-10 dataset consists of 60000x32 x 32 colour images divided in 10 classes, with 6000 images in each class. There are 50000 training images and 10000 test images.
We will use Keras with TensorFlow during the training of the model.
We will then output a random set of images in the form of 2 rows and 5 column with their corresponding predicted class name, probability and the true value of that image.

The main focus of this project is on how to apply CNN in real life using python, to learn more about CNN.

Introduction

In neural networks, Convolutional neural network (CNNs) is one of the main categories for image recognition, image classifications, Objects detections, recognizing faces etc., are some of the areas where CNNs are widely used. The practical benefit is that having fewer parameters greatly improves the time it takes to learn as well as reduces the amount of data required to train the model. Instead of a fully connected network of weights from each pixel, a CNN has just enough weights to look at a small patch of the image.

The architecture of a ConvNet is analogous to that of the connectivity pattern of Neurons in the Human Brain and was inspired by the organization of the Visual Cortex. Individual neurons respond to stimuli only in a restricted region of the visual field known as the Receptive Field. A collection of such fields overlap to cover the entire visual area.

The pre-processing required in a ConvNet is much lower as compared to other classification algorithms. While in primitive methods filters are hand-engineered, with enough training, ConvNets have the ability to learn these filters/characteristics.

Dataset

CIFAR is an acronym that stands for the Canadian Institute for Advanced Research and the CIFAR-10 dataset was developed along with the CIFAR-100 dataset by researchers at the CIFAR institute The classes are mutually exclusive and there is no overlap between them.
The dataset is comprised of 60,000 32×32-pixel color photographs of objects from 10 classes, such as frogs, birds, cats, ships, etc. The class labels and their standard associated integer values are listed below.
•	0: airplane
•	1: automobile
•	2: bird
•	3: cat
•	4: deer
•	5: dog
•	6: frog
•	7: horse
•	8: ship
•	9: truck

These are very small images, much smaller than a typical photograph, and the dataset was intended for computer vision research.
CIFAR-10 is a well-understood dataset and widely used for benchmarking computer vision algorithms in the field of machine learning. The problem is “solved.” It is relatively straightforward to achieve 70% classification accuracy.

The CIFAR10 dataset contains 60,000 colour images in 10 classes, with 6,000 images in each class. The dataset is divided into five training batches of 50,000 training images and one test batch with 10,000 testing images. The test batch contains exactly 1000 randomly-selected images from each class.

Platform

Colaboratory is a research tool for machine learning education and research. It’s a Jupyter notebook environment that requires no setup to use. Colaboratory works with most major browsers, and is most thoroughly tested with latest versions of Chrome, Firefox and Safari. All Colaboratory notebooks are stored in Google Drive. 
                         
Colaboratory notebooks can be shared just as you would with Google Docs or Sheets. Simply click the Share button at the top right of any Colaboratory notebook, or follow these Google Drive file sharing instructions.
Data Pre-processing

Now check shape of dataset and make a list of classes.
Next normalize the training dataset’s pixel values to be between 0 and 1. Now convert class labels to one-hot encoded vectors.

CNN Model

Our input image is a tensor whose width is 32 pixels and height is 32 pixels with 3 channels representing RGB(red, green, blue) color intensities. Thus we need to define a model which takes (None, 32, 32, 3) input shape and predicts (None, 10) output with probabilities for all classes. None in shapes stands for batch. We can do this by passing the input shape argument to our first layer.

We will Stack 3 convolutional layers with kernel size (3, 3) with growing number of filters (32, 32, 64) with padding size of same so input and output images will have same dimensions. We will add 2x2 pooling layer after every convolutional layer (conv-pool scheme).

To complete our model, you will feed the last output tensor from the convolutional base (of shape (4, 4, 64)) into one or more Dense layers to perform classification. But we have to use Flatten layer before first dense layer to reshape input volume into a flat vector.

Train Model

Now define model architecture and then assign loss function, optimizer, and metrics in our model by using model.compile( ) method.
Then we will initialize the number of epochs and use model.fit( ) function to start the training of our model by passing training data  and validation data. 
Initially we will train our model for 10 epochs and then gradually increase the number of epochs.

Outcome

While training the model with 10 epochs we have achieved maximum accuracy of 78% on training data and 71% accuracy on test data.
Output graph of the validation accuracy and validation loss.

Test Model

Now predict the probabilities of images in our test set using the model.predict( ) function, then we will get the name of the classes with maximum probability using argmax function and get the corresponding probabilities using the np.max( ).

Then we will output a random set of images in the form of 2 rows and 5 column with their corresponding predicted class name, probability and the true value of that image.

Conclusion

Our CNN model has achieved an accuracy of 71% on validation set with only 10 epochs.
But as we gradually increase the number of epochs the model achieves max accuracy of 82% on training data and 76% on validation data.


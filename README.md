# MNIST-Classification-with-Numpy-and-Backpropagation
Classifying MNIST digits without abstractifying the mathematical optimization involved in deep learning, diving into the meat of backpropagation.

![mnist](http://theanets.readthedocs.io/en/stable/_images/mnist-digits-small.png)

## Mathematical Optimization
Machine Learning uses mathematical optimization to minimize a loss, or measure of how bad the model is, where we then can descend the gradients of the parameters, called gradient descent. Finding the partial derivative, or rate of change, of a function with respect to its parameters is finding the gradient of those parameters.

## Deep neural networks
Multilayer feedforward networks contain input neurons, hidden neurons, and output neurons, which can be expressed as single numbers. Each and every input neuron affects each hidden neuron in the second layer, and so on until the output layer.



Each neuron is connected to a value in the next layer by a multiplier, or 'weight', and then added to a bias. This value, later expressed as 'z', is then sent through an activation function, such as sigmoid, to squish the value to the range of (0, 1), which is known as the activation. The activation of a neuron in any layer except for the input layer may be expressed as this:



where the superscript index is the layer, and the subscript index is of which neuron it is in a layer. However, this can be simplified significantly by expressing this as a series of matrix operations on an entire layer:




## Batches
Deep neural networks are usually trained on mini-batches of data at a time, finding an optimal balance between training time and results, where multiple inputs are fed in and multiple outputs are expected. This can be done by simply adding more rows to the input matrix:



## Backpropagation
The goal of backpropagation is to adjust the weights and biases in order to optimally classify MNIST digits. We first have to have a measure of how bad the network performs, so that we can minimize it using gradient descent. An easy yet effective one to implement is the Mean Squared Error: 



'Y' is the expected values, in our case that would be a vector of 0's except for the one that the input digit is, which will be a 1. 'Y hat' is what the network predicts what the expected labels should be, computed through the layers of matrix operations.
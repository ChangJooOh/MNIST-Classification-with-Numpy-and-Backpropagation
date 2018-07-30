# MNIST Classification with Numpy and Backpropagation
Classifying MNIST digits without abstractifying the mathematical optimization involved in deep learning, diving into the meat of backpropagation.

![mnist](http://theanets.readthedocs.io/en/stable/_images/mnist-digits-small.png)

## Mathematical Optimization
Machine Learning uses mathematical optimization to minimize a loss, or measure of how bad the model is, where we then can descend the gradients of the parameters, called gradient descent. Finding the partial derivative, or rate of change, of a function with respect to its parameters is finding the gradient of those parameters.

## Deep neural networks
Multilayer feedforward networks contain input neurons, hidden neurons, and output neurons, which can be expressed as single numbers. Each and every input neuron affects each hidden neuron in the second layer, and so on until the output layer.

![a](http://neuralnetworksanddeeplearning.com/images/tikz11.png)

Each neuron is connected to a value in the next layer by a multiplier, or 'weight', and then added to a bias. This value, later expressed as 'z', is then sent through an activation function, such as sigmoid, to squish the value to the range of (0, 1), which is known as the activation. The activation of a neuron in any layer except for the input layer may be expressed as this:

![b](https://latex.codecogs.com/png.latex?%5Clarge%20a_n%5EL%20%3D%20%5Csigma%20%28%28%5Csum_i%5Emw_i_n%5ELa_i%5EL%5E-%5E1%29%20&plus;%20b_n%5EL%29)

where the superscript index is the layer, and the subscript index is of which neuron it is in a layer. However, this can be simplified significantly by expressing this as a series of matrix operations on an entire layer:

![c](https://latex.codecogs.com/png.latex?a%5EL%20%3D%20%5Csigma%20%28%20%5Cbegin%7Bbmatrix%7D%20a_1%5EL%5E-%5E1%20%26.%20%26.%20%26%20a_m%5EL%5E-%5E1%20%5Cend%7Bbmatrix%7D%20%5Cbegin%7Bbmatrix%7D%20w_1_1%5EL%20%26.%20%26%20w_1_n%5EL%5C%5C%20.%20%26%20.%20%26%20.%5C%5C%20.%26.%20%26.%20%5C%5C%20w_m_1%5EL%20%26%20.%20%26%20w_m_n%5EL%20%5Cend%7Bbmatrix%7D%20&plus;%20%5Cbegin%7Bbmatrix%7D%20b_1%5EL%20%26.%20%26%20b_n%5EL%20%5Cend%7Bbmatrix%7D%20%29)

![d](https://latex.codecogs.com/png.latex?a%5EL%20%3D%20%5Csigma%20%28%20a%5EL%5E-%5E1w%5EL%20&plus;%20b%5EL%20%29)

## Batches
Deep neural networks are usually trained on mini-batches of data at a time, finding an optimal balance between training time and results, where multiple inputs are fed in and multiple outputs are expected. This can be done by simply adding more rows to the input matrix.

## Backpropagation
The goal of backpropagation is to adjust the weights and biases in order to optimally classify MNIST digits. We first have to have a measure of how bad the network performs, so that we can minimize it using gradient descent. An easy yet effective one to implement is the Mean Squared Error: 

![e](https://latex.codecogs.com/gif.latex?MSE%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%5Csum_i%5Em%28y-a%5EL%29%5E2)

'Y' is the expected values, in our case that would be a vector of 0's except for the index that is the handwritten input digit, which will be a 1. Where L is the number of layers, so a superscript L would be the final activations, and is what the network predicts what the expected labels should be, computed through the layers of matrix operations.

To do this we must find the partial derivative with respect for each weight and bias, or gradients, using the chain rule. These are the calculated derivatives of a simple neural network with one input neuron, and one output neuron. Remember that 'z' is the pre-activation.

![f](https://latex.codecogs.com/png.latex?%5Cfrac%7B%5Cpartial%20Loss%7D%7B%5Cpartial%20w%5EL%7D%20%3D%20%5Cfrac%7B%5Cpartial%20Loss%7D%7B%5Cpartial%20a%5EL%7D%20%5Cfrac%7B%5Cpartial%20a%5EL%7D%7B%5Cpartial%20z%5EL%7D%20%5Cfrac%7B%5Cpartial%20z%5EL%7D%7B%5Cpartial%20w%5EL%7D)

![g](https://latex.codecogs.com/png.latex?%5Cfrac%7B%5Cpartial%20Loss%7D%7B%5Cpartial%20w%5EL%7D%20%3D%20%28y%20-%20a%5EL%29%7B%5Csigma%20%7D%27%28z%5EL%29a%5E%7BL-1%7D)

![i](https://latex.codecogs.com/png.latex?%5Cfrac%7B%5Cpartial%20Loss%7D%7B%5Cpartial%20b%5EL%7D%20%3D%20%5Cfrac%7B%5Cpartial%20Loss%7D%7B%5Cpartial%20a%5EL%7D%20%5Cfrac%7B%5Cpartial%20a%5EL%7D%7B%5Cpartial%20z%5EL%7D%20%5Cfrac%7B%5Cpartial%20z%5EL%7D%7B%5Cpartial%20b%5EL%7D)

![j](https://latex.codecogs.com/png.latex?%5Cfrac%7B%5Cpartial%20Loss%7D%7B%5Cpartial%20b%5EL%7D%20%3D%20%28y%20-%20a%5EL%29%7B%5Csigma%20%7D%27%28z%5EL%29)

If we add another layer of one neuron to the input, how do we find the weight and bias gradient of the first layer? We have to find the gradient of the activation in the second layer, and use it in the chain rule the same way we used the derivative of our loss function.

![k](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20Loss%7D%7B%5Cpartial%20a%5E%7BL-1%7D%7D%3D%20%5Cfrac%7B%5Cpartial%20Loss%7D%7B%5Cpartial%20a%5E%7BL%7D%7D%20%5Cfrac%7B%5Cpartial%20a%5E%7BL%7D%7D%7B%5Cpartial%20z%5E%7BL%7D%7D%20%5Cfrac%7B%5Cpartial%20z%5E%7BL%7D%7D%7B%5Cpartial%20a%5E%7BL-1%7D%7D)

![l](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20Loss%7D%7B%5Cpartial%20a%5E%7BL-1%7D%7D%3D%20%28y-a%5EL%29%7B%5Csigma%7D%27%28z%5EL%29w%5EL)

![m](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20Loss%7D%7B%5Cpartial%20w%5E%7BL-1%7D%7D%3D%20%5Cfrac%7B%5Cpartial%20Loss%7D%7B%5Cpartial%20a%5E%7BL-1%7D%7D%7B%5Csigma%7D%27%28z%5E%7BL-1%7D%29a%5E%7BL-2%7D)

![n](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20Loss%7D%7B%5Cpartial%20b%5E%7BL-1%7D%7D%3D%20%5Cfrac%7B%5Cpartial%20Loss%7D%7B%5Cpartial%20a%5E%7BL-1%7D%7D%7B%5Csigma%7D%27%28z%5E%7BL-1%7D%29)

When we add more neurons per layer, we can abstractify the backpropagation process in the same way we did with forward propagation: using matrix operations. The math does not change that much. Representing transpose matrices conflict with the superscript, so the layer it is in is now subscript.

![o](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20Loss%7D%7B%5Cpartial%20w_%7BL%7D%7D%3D%20a%5ET_%7BL-1%7D%28y-a_L%29%7B%5Csigma%7D%27%28z_%7BL%7D%29)

![p](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20Loss%7D%7B%5Cpartial%20b_%7BL%7D%7D%3D%20%28y-a_L%29%7B%5Csigma%7D%27%28z_%7BL%7D%29)

![q](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20Loss%7D%7B%5Cpartial%20a_%7BL-1%7D%7D%3D%20%28y-a_L%29%7B%5Csigma%7D%27%28z_%7BL%7D%29w%5ET_L)

Thanks to Online LaTeX Equations for a great equation editor, and 3Blue1Brown for the math!

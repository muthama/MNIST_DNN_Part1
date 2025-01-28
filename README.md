# MNIST_DNN_Part1

## Repository Description
This repository contains the implementation of a deep neural network (DNN) for handwritten digit recognition, developed from scratch in C++. The project is part of an academic assignment that demonstrates the design and training of a vanilla DNN to classify digits from the MNIST dataset without the use of any external machine learning libraries. 

The implementation strictly follows the configuration described in the paper:
> Dan Claudiu Ciresan, et al, "Deep Big Simple Neural Nets Excel on Hand-written Digit Recognition," arXiv:1003.0358, Mar, 2010. ([arXiv Link](https://arxiv.org/pdf/1003.0358))

The repository also includes the necessary file structure to organize the MNIST dataset and train the network effectively.

---

## Features
- Implementation of a feedforward multi-layer perceptron (MLP) with backpropagation.
- Custom activation functions, including a scaled hyperbolic tangent function.
- Training and evaluation of the model using the MNIST dataset.
- High accuracy achieved on a subset of the MNIST digits.

---

## Implementation Details
The code implements a deep neural network with several key components:

### Data Processing
The implementation begins with careful preprocessing of the MNIST dataset. The system reads binary IDX files containing both images and labels, performing necessary normalization steps. Image pixel values are scaled to the range [-1, 1] to improve training stability, while labels are processed as integers from 0 to 9.

### Neural Network Architecture
The network is structured as a Multi-Layer Perceptron (MLP) with three layers, each defined by its weights (W) and biases (b). The architecture includes:
- Input layer to first hidden layer: W1, b1
- First to second hidden layer: W2, b2
- Second hidden layer to output layer: W3, b3

### Activation Functions
A specialized scaled hyperbolic tangent function is employed as the activation function, using a scaling factor of 1.7159. This scaling was carefully chosen to help prevent neuron saturation during training, with the implementation including both the forward function and its derivative for backpropagation.

### Training Process
The training implementation follows these steps:
1. Forward propagation through the network layers, culminating in a softmax activation for classification
2. Loss computation using cross-entropy
3. Backpropagation to update network weights
4. Prediction generation by selecting the highest probability class

The training loop iterates through a configured number of epochs, processing the entire training dataset and regularly evaluating performance on the test set to monitor progress.

---

## File Structure
```
MNIST_DNN_Part1/
├── mnist_dnn.cpp    # Main program file
├── data/                # Directory for MNIST dataset files
│   ├── train-images-idx3-ubyte
│   ├── train-labels-idx1-ubyte
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
├── CMakeLists.txt       # Build configuration for the project
└── README.md            # Project documentation
```

---

## Getting Started

### Prerequisites
- **C++ Compiler:** GCC 7.5+, Clang, or MSVC
- **CMake:** Version 3.10+
- MNIST dataset files placed in the `data` directory

### Build and Run

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/MNIST_DNN_Part1.git
   cd MNIST_DNN_Part1
   ```

2. Create a build directory and configure the project:
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

3. Run the program:
   ```bash
   ./MNIST_DNN_Part1 data/train-images-idx3-ubyte data/train-labels-idx1-ubyte \
                      data/t10k-images-idx3-ubyte data/t10k-labels-idx1-ubyte
   ```

---

## Dataset
The MNIST dataset can be downloaded from [DeepAI](https://deepai.org/dataset/mnist). Place the files in the `data/` directory as shown in the file structure above.

---

## License
This project is released under the MIT License.

---

## References
1. Dan Claudiu Ciresan, et al, "Deep Big Simple Neural Nets Excel on Hand-written Digit Recognition," arXiv:1003.0358, Mar, 2010.
2. [MNIST Dataset on DeepAI](https://deepai.org/dataset/mnist)

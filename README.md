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

## Experimental Results and Analysis

The network was trained for 10 epochs, demonstrating impressive performance improvements over the training period. Here are the detailed results and their interpretation:

### Performance Metrics

```
Epoch 1 - Training Accuracy: 86.7133%, Loss (not real CE): 7972
         Test Accuracy: 92.16%
Epoch 2 - Training Accuracy: 93.815%, Loss (not real CE): 3711
         Test Accuracy: 94.43%
Epoch 3 - Training Accuracy: 95.5367%, Loss (not real CE): 2678
         Test Accuracy: 95.38%
Epoch 4 - Training Accuracy: 96.4367%, Loss (not real CE): 2138
         Test Accuracy: 96.1%
Epoch 5 - Training Accuracy: 97.055%, Loss (not real CE): 1767
         Test Accuracy: 96.64%
Epoch 6 - Training Accuracy: 97.4767%, Loss (not real CE): 1514
         Test Accuracy: 96.93%
Epoch 7 - Training Accuracy: 97.8667%, Loss (not real CE): 1280
         Test Accuracy: 97.08%
Epoch 8 - Training Accuracy: 98.1333%, Loss (not real CE): 1120
         Test Accuracy: 97.21%
Epoch 9 - Training Accuracy: 98.37%, Loss (not real CE): 978
         Test Accuracy: 97.25%
Epoch 10 - Training Accuracy: 98.615%, Loss (not real CE): 831
         Test Accuracy: 97.16%
```

### Analysis of Results

#### Training Accuracy
The network demonstrates strong learning capabilities, with training accuracy improving from 86.7% to 98.6% over ten epochs. This substantial improvement indicates that the network successfully learns to recognize patterns in the training data. The steady increase in accuracy, rather than sudden jumps, suggests stable and consistent learning throughout the training process.

#### Loss Metric
The loss value, while not implementing traditional cross-entropy, serves as a practical measure of model performance. It represents the count of misclassifications during training, decreasing significantly from 7,972 to 831 over the training period. This dramatic reduction in classification errors aligns with the observed improvement in accuracy and confirms the network's learning progression.

#### Test Accuracy
Perhaps the most significant indicator of the model's true performance is its test accuracy, which improves from 92.16% to 97.16%. This metric is particularly important as it represents the network's ability to generalize to unseen data. The test accuracy's steady increase, nearly matching the training accuracy's trajectory, indicates that the network is learning meaningful features rather than merely memorizing the training data.

The final test accuracy of 97.16% is particularly impressive for a vanilla neural network implementation, approaching the performance levels of more complex architectures while maintaining simplicity in design and implementation.

### Learning Dynamics
The results reveal several interesting aspects of the network's learning process:
1. The most dramatic improvements occur in the early epochs, with the first two epochs showing the largest accuracy gains.
2. The learning curve begins to plateau around epoch 8, suggesting the network is approaching its optimal performance for the current architecture.
3. The small gap between training and test accuracy (approximately 1.5%) indicates good generalization without significant overfitting.

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

# MNIST_DNN_Part1

## Repository Description
This repository contains the implementation of a deep neural network (DNN) for handwritten digit recognition, developed from scratch in C++. The project demonstrates the fundamental principles of neural networks through a vanilla implementation that classifies digits from the MNIST dataset without using external machine learning libraries.

The implementation follows the architecture and methodology described in:
> Dan Claudiu Ciresan, et al, "Deep Big Simple Neural Nets Excel on Hand-written Digit Recognition," arXiv:1003.0358, Mar, 2010.

---

## Features
- Complete feedforward neural network implementation with backpropagation
- Custom activation functions optimized for training stability
- MNIST dataset processing and normalization
- High-accuracy digit classification
- Educational implementation focusing on clarity and fundamentals

---

## File Structure
```
MNIST_DNN_Part1/
├── mnist_dnn.cpp         # Main program file
├── data/                 # Directory for MNIST dataset files
│   ├── train-images-idx3-ubyte
│   ├── train-labels-idx1-ubyte
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
├── CMakeLists.txt        # Build configuration for the project
└── README.md             # Project documentation
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

## Implementation Details

### Network Architecture
The implementation uses a three-layer neural network design:
1. **Input Layer**: 784 neurons (28x28 pixel images)
2. **Hidden Layer 1**: 128 neurons with scaled tanh activation
3. **Hidden Layer 2**: 64 neurons with scaled tanh activation
4. **Output Layer**: 10 neurons (one per digit) with softmax activation

### Core Components

1. **Data Preprocessing**
   - Binary IDX file parsing for MNIST format
   - Input normalization to [-1, 1] range
   - Label processing for classification

2. **Activation Functions**
   - Scaled hyperbolic tangent (factor: 1.7159)
   - Custom derivative implementation for backpropagation
   - Softmax for output layer classification

3. **Training Process**
   - Forward propagation through layers
   - Cross-entropy loss computation
   - Backpropagation for weight updates
   - Batch processing with error accumulation

---

## Training Results and Analysis

### Performance Metrics

```
Epoch 1 - Training Accuracy: 86.7133%, Loss: 7972
         Test Accuracy: 92.16%
Epoch 2 - Training Accuracy: 93.815%, Loss: 3711
         Test Accuracy: 94.43%
Epoch 3 - Training Accuracy: 95.5367%, Loss: 2678
         Test Accuracy: 95.38%
Epoch 4 - Training Accuracy: 96.4367%, Loss: 2138
         Test Accuracy: 96.1%
Epoch 5 - Training Accuracy: 97.055%, Loss: 1767
         Test Accuracy: 96.64%
Epoch 6 - Training Accuracy: 97.4767%, Loss: 1514
         Test Accuracy: 96.93%
Epoch 7 - Training Accuracy: 97.8667%, Loss: 1280
         Test Accuracy: 97.08%
Epoch 8 - Training Accuracy: 98.1333%, Loss: 1120
         Test Accuracy: 97.21%
Epoch 9 - Training Accuracy: 98.37%, Loss: 978
         Test Accuracy: 97.25%
Epoch 10 - Training Accuracy: 98.615%, Loss: 831
         Test Accuracy: 97.16%
```

### Result Analysis

The training results demonstrate three distinct phases of learning:

1. **Initial Learning Phase (Epochs 1-3)**
   - Rapid improvement in training accuracy from 86.7% to 95.5%
   - Test accuracy shows strong gains from 92.16% to 95.38%
   - Dramatic reduction in loss from 7972 to 2678
   - Network quickly learns primary digit features

2. **Refinement Phase (Epochs 4-7)**
   - Steady improvement in training accuracy from 96.4% to 97.8%
   - Test accuracy maintains close correlation, reaching 97.08%
   - Loss continues to decrease but at a slower rate
   - Network fine-tunes feature recognition

3. **Convergence Phase (Epochs 8-10)**
   - Training accuracy approaches 98.6%
   - Test accuracy stabilizes around 97.2%
   - Loss reduction slows significantly
   - Network reaches optimal performance for its architecture

### Learning Dynamics

Several key observations about the network's learning process:

1. **Generalization Performance**
   - Small gap between training and test accuracy (≈1.5%)
   - Indicates good generalization without overfitting
   - Validates the network architecture and hyperparameters

2. **Learning Stability**
   - Consistent improvement across all metrics
   - No significant fluctuations or regression
   - Demonstrates robust training process

3. **Final Performance**
   - 97.16% test accuracy represents strong performance for a vanilla implementation
   - Comparable to results reported in literature for similar architectures
   - Demonstrates effectiveness of the chosen activation functions and network design

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

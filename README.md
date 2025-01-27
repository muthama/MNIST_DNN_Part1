# MNIST_DNN_Part1

## Repository Description
This repository contains the implementation of a deep neural network (DNN) for handwritten digit recognition, developed from scratch in C++. The project is part of an academic assignment that demonstrates the design and training of a vanilla DNN to classify digits from the MNIST dataset without the use of any external machine learning libraries. 

The implementation strictly follows the configuration described in the paper:
> Dan Claudiu Ciresan, et al, “Deep Big Simple Neural Nets Excel on Hand-written Digit Recognition,” arXiv:1003.0358, Mar, 2010. ([arXiv Link](https://arxiv.org/pdf/1003.0358))

The repository also includes the necessary file structure to organize the MNIST dataset and train the network effectively.

---

## Features
- Implementation of a feedforward multi-layer perceptron (MLP) with backpropagation.
- Custom activation functions, including a scaled hyperbolic tangent function.
- Training and evaluation of the model using the MNIST dataset.
- High accuracy achieved on a subset of the MNIST digits.

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
1. Dan Claudiu Ciresan, et al, “Deep Big Simple Neural Nets Excel on Hand-written Digit Recognition,” arXiv:1003.0358, Mar, 2010.
2. [MNIST Dataset on DeepAI](https://deepai.org/dataset/mnist)

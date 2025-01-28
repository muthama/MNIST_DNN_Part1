/******************************************************************************
 * Compile example:
 *   g++ -O2 -std=c++11 mnist_dnn.cpp -o mnist_dnn
 *
 * Example run:
 *   ./mnist_dnn train-images-idx3-ubyte train-labels-idx1-ubyte \
 *               t10k-images-idx3-ubyte t10k-labels-idx1-ubyte
 *
 * Note:
 *  - Make sure the file paths are correct for your MNIST dataset.
 *  - This code uses a simpler MLP architecture: 784 -> 128 -> 64 -> 10
 ******************************************************************************/

#include <iostream>   // For console I/O
#include <fstream>    // For file operations (reading MNIST files)
#include <vector>     // For storing images, weights, etc.
#include <cmath>      // For math functions (exp, tanh, etc.)
#include <cstdlib>    // For std::rand(), std::srand()
#include <cstring>    // For string operations (optional)
#include <algorithm>  // For std::max_element, etc.

// ----------------------------- Configuration -------------------------------- //
static const int IMAGE_SIZE = 28;       // Each MNIST image is 28 x 28
static const int INPUT_SIZE = 28 * 28;  // Flattened input dimension: 784
static const int NUM_CLASSES = 10;      // Number of digit classes (0 through 9)

// Simple network architecture for demonstration: 784 -> 128 -> 64 -> 10
static const int HIDDEN1 = 128;
static const int HIDDEN2 = 64;

// Hyperparameters for training
static const float LEARNING_RATE = 0.001f;  // How fast the network updates weights
static const int EPOCHS = 10;               // Number of training epochs
static const int TRAINING_SAMPLES = 60000;  // Number of images in full MNIST training set
static const int TEST_SAMPLES = 10000;      // Number of images in full MNIST test set

// --------------------- MNIST Reading Utilities ------------------------------ //
/**
 * @brief Reverses the endianness of a 4-byte integer.
 *
 * MNIST IDX files store multi-byte values in big-endian format;
 * on many systems (little-endian), we need to swap the byte order.
 */
int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;         // lowest byte
    c2 = (i >> 8) & 255;  // 2nd byte
    c3 = (i >> 16) & 255; // 3rd byte
    c4 = (i >> 24) & 255; // highest byte
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

/**
 * @brief Reads MNIST images from an IDX file and stores them in a 2D vector.
 *
 * @param filename Path to the MNIST "images" file.
 * @param images   Vector of image data to fill. Each entry is a vector of 784 floats.
 * @param count    Number of images to read from the file.
 * @return true if successful, false otherwise.
 */
bool readMNISTImages(const std::string &filename,
                     std::vector<std::vector<float> > &images,
                     int count)
{
    std::ifstream file(filename, std::ios::binary); // open file in binary mode
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filename << std::endl;
        return false;
    }

    // Read the metadata (magic number, number of images, rows, cols) in 32-bit integers
    int magicNumber = 0;
    int numberOfImages = 0;
    int rows = 0;
    int cols = 0;

    file.read(reinterpret_cast<char *>(&magicNumber), sizeof(magicNumber));
    file.read(reinterpret_cast<char *>(&numberOfImages), sizeof(numberOfImages));
    file.read(reinterpret_cast<char *>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char *>(&cols), sizeof(cols));

    // Convert from big-endian to native endianness
    magicNumber = reverseInt(magicNumber);
    numberOfImages = reverseInt(numberOfImages);
    rows = reverseInt(rows);
    cols = reverseInt(cols);

    // Safety check if the file doesn't have enough images
    if (numberOfImages < count) {
        std::cerr << "File contains fewer images (" << numberOfImages
                  << ") than required (" << count << "). Using only "
                  << numberOfImages << " images." << std::endl;
        count = numberOfImages;
    }

    // Resize the images vector to hold 'count' images, each with 'rows*cols' pixels
    images.resize(count, std::vector<float>(rows * cols, 0.0f));

    // Read each image from the file
    for (int i = 0; i < count; ++i) {
        for (int r = 0; r < rows * cols; ++r) {
            unsigned char pixel = 0;
            file.read(reinterpret_cast<char *>(&pixel), sizeof(pixel));
            // Normalize pixel values to the range [-1, 1] instead of [0,1]
            images[i][r] = (pixel / 255.0f) * 2.0f - 1.0f;
        }
    }

    file.close();
    return true;
}

/**
 * @brief Reads MNIST labels from an IDX file and stores them in a vector of integers.
 *
 * @param filename Path to the MNIST "labels" file.
 * @param labels   Vector of labels to fill.
 * @param count    Number of labels to read.
 * @return true if successful, false otherwise.
 */
bool readMNISTLabels(const std::string &filename,
                     std::vector<int> &labels,
                     int count)
{
    std::ifstream file(filename, std::ios::binary); // open file in binary mode
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filename << std::endl;
        return false;
    }

    // Read metadata (magic number, number of labels)
    int magicNumber = 0;
    int numberOfLabels = 0;

    file.read(reinterpret_cast<char *>(&magicNumber), sizeof(magicNumber));
    file.read(reinterpret_cast<char *>(&numberOfLabels), sizeof(numberOfLabels));
    magicNumber = reverseInt(magicNumber);
    numberOfLabels = reverseInt(numberOfLabels);

    // Safety check if the file doesn't have enough labels
    if (numberOfLabels < count) {
        std::cerr << "File contains fewer labels (" << numberOfLabels
                  << ") than required (" << count << "). Using only "
                  << numberOfLabels << " labels." << std::endl;
        count = numberOfLabels;
    }

    // Resize the labels vector to hold 'count' labels
    labels.resize(count);

    // Read each label from the file
    for (int i = 0; i < count; ++i) {
        unsigned char label = 0;
        file.read(reinterpret_cast<char *>(&label), 1);
        labels[i] = static_cast<int>(label);
    }

    file.close();
    return true;
}

// --------------------- Activation (Scaled Tanh) ---------------------------- //
/**
 * @brief Scaled Tanh activation function.
 *
 * This function applies 1.7159 * tanh((2/3) * x).
 * Scaling helps avoid saturation and can improve training stability.
 */
inline float scaledTanh(float x) {
    const float alpha = 1.7159f;
    const float beta = 2.0f / 3.0f;
    return alpha * std::tanh(beta * x);
}

/**
 * @brief Derivative of the scaled Tanh activation function.
 *
 * If y = 1.7159 * tanh((2/3)*x), then
 * dy/dx = 1.7159 * (2/3) * sech^2((2/3)*x).
 */
inline float scaledTanhDerivative(float x) {
    const float alpha = 1.7159f;
    const float beta = 2.0f / 3.0f;
    float th = std::tanh(beta * x);   // tanh((2/3)*x)
    float sech2 = 1.0f - th * th;     // sech^2(z) = 1 - tanh^2(z)
    return alpha * beta * sech2;
}

// -------------------- Helper: Random Initialization ------------------------- //
/**
 * @brief Generates a random weight in the range [-0.05, 0.05].
 *
 * Used to initialize the network's weights so that they start near zero.
 */
inline float randWeight() {
    return ((std::rand() / static_cast<float>(RAND_MAX)) * 0.1f) - 0.05f;
}

// -------------------- MLP Class Definition --------------------------------- //
/**
 * @class MLP
 * @brief A simple Multi-Layer Perceptron with two hidden layers.
 *
 * Architecture:
 *   Input (784) -> Hidden1 (128) -> Hidden2 (64) -> Output (10)
 */
class MLP {
public:
    /**
     * @brief Constructor that allocates and initializes weights and biases.
     */
    MLP() {
        // Allocate W1 and b1 (Input -> Hidden1)
        W1.resize(INPUT_SIZE, std::vector<float>(HIDDEN1));
        b1.resize(HIDDEN1);

        // Allocate W2 and b2 (Hidden1 -> Hidden2)
        W2.resize(HIDDEN1, std::vector<float>(HIDDEN2));
        b2.resize(HIDDEN2);

        // Allocate W3 and b3 (Hidden2 -> Output)
        W3.resize(HIDDEN2, std::vector<float>(NUM_CLASSES));
        b3.resize(NUM_CLASSES);

        // Randomly initialize W1 and b1
        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < HIDDEN1; ++j) {
                W1[i][j] = randWeight();
            }
        }
        for (int j = 0; j < HIDDEN1; ++j) {
            b1[j] = randWeight();
        }

        // Randomly initialize W2 and b2
        for (int i = 0; i < HIDDEN1; ++i) {
            for (int j = 0; j < HIDDEN2; ++j) {
                W2[i][j] = randWeight();
            }
        }
        for (int j = 0; j < HIDDEN2; ++j) {
            b2[j] = randWeight();
        }

        // Randomly initialize W3 and b3
        for (int i = 0; i < HIDDEN2; ++i) {
            for (int j = 0; j < NUM_CLASSES; ++j) {
                W3[i][j] = randWeight();
            }
        }
        for (int j = 0; j < NUM_CLASSES; ++j) {
            b3[j] = randWeight();
        }
    }

    /**
     * @brief Forward pass for a single input sample (feedforward).
     *
     * @param x A vector of 784 floats (the normalized input image).
     */
    void forward(const std::vector<float> &x) {
        // 1) Input -> Hidden1
        z1.resize(HIDDEN1);
        a1.resize(HIDDEN1);
        for (int j = 0; j < HIDDEN1; ++j) {
            float sum = b1[j]; // start with bias
            for (int i = 0; i < INPUT_SIZE; ++i) {
                sum += x[i] * W1[i][j];  // weighted sum of inputs
            }
            z1[j] = sum;                // pre-activation value
            a1[j] = scaledTanh(sum);    // apply scaled tanh activation
        }

        // 2) Hidden1 -> Hidden2
        z2.resize(HIDDEN2);
        a2.resize(HIDDEN2);
        for (int j = 0; j < HIDDEN2; ++j) {
            float sum = b2[j]; // bias
            for (int i = 0; i < HIDDEN1; ++i) {
                sum += a1[i] * W2[i][j]; // weighted sum of hidden1 activations
            }
            z2[j] = sum;
            a2[j] = scaledTanh(sum);    // activation for layer2
        }

        // 3) Hidden2 -> Output (logits)
        z3.resize(NUM_CLASSES);
        a3.resize(NUM_CLASSES);         // final output probabilities (after softmax)
        for (int j = 0; j < NUM_CLASSES; ++j) {
            float sum = b3[j];          // bias
            for (int i = 0; i < HIDDEN2; ++i) {
                sum += a2[i] * W3[i][j]; // weighted sum of hidden2 activations
            }
            z3[j] = sum;                // pre-softmax value
        }

        // Apply softmax to get probabilities for each class
        float maxLogit = *std::max_element(z3.begin(), z3.end());
        float sumExp = 0.0f;

        // Compute exponentials (shifted by maxLogit to avoid numeric overflow)
        for (int j = 0; j < NUM_CLASSES; ++j) {
            float e = std::exp(z3[j] - maxLogit);
            a3[j] = e;
            sumExp += e;
        }

        // Normalize to get the final probabilities
        for (int j = 0; j < NUM_CLASSES; ++j) {
            a3[j] /= sumExp;
        }
    }

    /**
     * @brief Backpropagation for a single training sample (with cross-entropy loss).
     *
     * @param x     The input image (784 floats).
     * @param label The correct class label (0-9).
     */
    void backward(const std::vector<float> &x, int label) {
        // 1) Compute output layer delta = (predicted_prob - target)
        std::vector<float> delta3(NUM_CLASSES);
        for (int j = 0; j < NUM_CLASSES; ++j) {
            float target = (j == label) ? 1.0f : 0.0f; // One-hot target
            delta3[j] = a3[j] - target;                // derivative of cross-entropy + softmax
        }

        // 2) Update Hidden2->Output weights/bias
        for (int j = 0; j < NUM_CLASSES; ++j) {
            for (int i = 0; i < HIDDEN2; ++i) {
                W3[i][j] -= LEARNING_RATE * delta3[j] * a2[i];
            }
            b3[j] -= LEARNING_RATE * delta3[j];
        }

        // 3) Calculate delta for the Hidden2 layer
        std::vector<float> delta2(HIDDEN2, 0.0f);
        for (int i = 0; i < HIDDEN2; ++i) {
            float grad = 0.0f;
            // Sum of contributions from each output neuron
            for (int j = 0; j < NUM_CLASSES; ++j) {
                grad += delta3[j] * W3[i][j];
            }
            // Multiply by derivative of scaled tanh at z2[i]
            grad *= scaledTanhDerivative(z2[i]);
            delta2[i] = grad;
        }

        // 4) Update Hidden1->Hidden2 weights/bias
        for (int j = 0; j < HIDDEN2; ++j) {
            for (int i = 0; i < HIDDEN1; ++i) {
                W2[i][j] -= LEARNING_RATE * delta2[j] * a1[i];
            }
            b2[j] -= LEARNING_RATE * delta2[j];
        }

        // 5) Calculate delta for the Hidden1 layer
        std::vector<float> delta1(HIDDEN1, 0.0f);
        for (int i = 0; i < HIDDEN1; ++i) {
            float grad = 0.0f;
            for (int j = 0; j < HIDDEN2; ++j) {
                grad += delta2[j] * W2[i][j];
            }
            grad *= scaledTanhDerivative(z1[i]);
            delta1[i] = grad;
        }

        // 6) Update Input->Hidden1 weights/bias
        for (int j = 0; j < HIDDEN1; ++j) {
            for (int i = 0; i < INPUT_SIZE; ++i) {
                W1[i][j] -= LEARNING_RATE * delta1[j] * x[i];
            }
            b1[j] -= LEARNING_RATE * delta1[j];
        }
    }

    /**
     * @brief Predicts the class label for a single input sample.
     *
     * @param x The input image (784 floats).
     * @return The predicted digit label (0-9).
     */
    int predict(const std::vector<float> &x) {
        // Run forward pass
        forward(x);

        // Find the class with the maximum probability in a3
        float maxVal = a3[0];
        int maxIdx = 0;
        for (int j = 1; j < NUM_CLASSES; ++j) {
            if (a3[j] > maxVal) {
                maxVal = a3[j];
                maxIdx = j;
            }
        }
        return maxIdx;
    }

private:
    // Weight matrices and bias vectors for each layer
    // Dimensions:
    //   W1: [784 x 128], b1: [128]
    //   W2: [128 x 64],  b2: [64]
    //   W3: [64 x 10],   b3: [10]
    std::vector<std::vector<float> > W1;
    std::vector<float> b1;
    std::vector<std::vector<float> > W2;
    std::vector<float> b2;
    std::vector<std::vector<float> > W3;
    std::vector<float> b3;

    // Intermediate (forward pass) results
    std::vector<float> z1, a1; // z1: pre-activations in layer1, a1: activations
    std::vector<float> z2, a2; // same for layer2
    std::vector<float> z3, a3; // same for output layer
};

// --------------------------- Main Program ----------------------------------- //
int main(int argc, char **argv) {
    // Basic argument check: we need 4 file paths
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <train_images> <train_labels> <test_images> <test_labels>\n";
        return 1;
    }

    // Parse command-line arguments for file paths
    std::string trainImagePath = argv[1];
    std::string trainLabelPath = argv[2];
    std::string testImagePath  = argv[3];
    std::string testLabelPath  = argv[4];

    // Seed the random generator for weight initialization (fixed for reproducibility)
    std::srand(123);

    // Read training data (images and labels)
    std::vector<std::vector<float> > trainImages;
    std::vector<int> trainLabels;
    if (!readMNISTImages(trainImagePath, trainImages, TRAINING_SAMPLES) ||
        !readMNISTLabels(trainLabelPath, trainLabels, TRAINING_SAMPLES))
    {
        std::cerr << "Error reading training data.\n";
        return 1;
    }

    // Read test data (images and labels)
    std::vector<std::vector<float> > testImages;
    std::vector<int> testLabels;
    if (!readMNISTImages(testImagePath, testImages, TEST_SAMPLES) ||
        !readMNISTLabels(testLabelPath, testLabels, TEST_SAMPLES))
    {
        std::cerr << "Error reading test data.\n";
        return 1;
    }

    // Create an instance of our MLP model
    MLP dnn;

    // Training loop for the specified number of epochs
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        // (Optional) Shuffle training data each epoch for better generalization
        // Here, it is skipped for brevity.

        float epochLoss = 0.0f;   // We'll track an approximate "loss"
        int correctCount = 0;     // Track how many training samples are classified correctly

        // Train on each sample
        for (int i = 0; i < TRAINING_SAMPLES; ++i) {
            // Forward pass
            dnn.forward(trainImages[i]);

            // For demonstration, we don't compute the real cross-entropy here.
            // Instead, we just log 0 or 1 depending on correctness (this is simplified).
            int pred = dnn.predict(trainImages[i]);
            if (pred == trainLabels[i]) {
                correctCount++;
            }

            // Backpropagate error
            dnn.backward(trainImages[i], trainLabels[i]);

            // Accumulate a dummy loss measure
            // (For real training, you'd do something like: -log(prob_of_correct_label))
            epochLoss += (pred == trainLabels[i]) ? 0.0f : 1.0f;
        }

        // Compute training accuracy
        float trainAccuracy = (100.0f * correctCount) / TRAINING_SAMPLES;
        std::cout << "Epoch " << (epoch + 1)
                  << " - Training Accuracy: " << trainAccuracy
                  << "%, Loss (not real CE): " << epochLoss
                  << std::endl;

        // Evaluate on the test set each epoch
        int testCorrect = 0;
        for (int i = 0; i < TEST_SAMPLES; ++i) {
            int pred = dnn.predict(testImages[i]);
            if (pred == testLabels[i]) {
                testCorrect++;
            }
        }
        float testAccuracy = (100.0f * testCorrect) / TEST_SAMPLES;
        std::cout << "         Test Accuracy: " << testAccuracy << "%\n";
    }

    return 0;
}

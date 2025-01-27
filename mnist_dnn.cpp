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

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>

// ----------------------------- Configuration -------------------------------- //
static const int IMAGE_SIZE = 28; // MNIST images are 28x28
static const int INPUT_SIZE = 28 * 28; // 784
static const int NUM_CLASSES = 10; // digits 0-9

// Choose a smaller architecture for illustration
// For a bigger network (like in the paper), expand as needed, e.g. 784->2500->2000->1500->10
static const int HIDDEN1 = 128;
static const int HIDDEN2 = 64;

// Hyperparameters
static const float LEARNING_RATE = 0.001f;
static const int EPOCHS = 10; // Increase to 100+ for higher accuracy
static const int TRAINING_SAMPLES = 60000; // Full MNIST training set
static const int TEST_SAMPLES = 10000; // Full MNIST test set

// --------------------- MNIST Reading Utilities ------------------------------ //
int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int) c1 << 24) + ((int) c2 << 16) + ((int) c3 << 8) + c4;
}

// Reads MNIST images from the IDX file
bool readMNISTImages(const std::string &filename, std::vector<std::vector<float> > &images, int count) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filename << std::endl;
        return false;
    }
    int magicNumber = 0;
    int numberOfImages = 0;
    int rows = 0;
    int cols = 0;

    file.read(reinterpret_cast<char *>(&magicNumber), sizeof(magicNumber));
    file.read(reinterpret_cast<char *>(&numberOfImages), sizeof(numberOfImages));
    file.read(reinterpret_cast<char *>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char *>(&cols), sizeof(cols));

    magicNumber = reverseInt(magicNumber);
    numberOfImages = reverseInt(numberOfImages);
    rows = reverseInt(rows);
    cols = reverseInt(cols);

    // Safety check
    if (numberOfImages < count) {
        std::cerr << "File contains fewer images (" << numberOfImages << ") than required ("
                << count << "). Using only " << numberOfImages << " images." << std::endl;
        count = numberOfImages;
    }

    // Resize the vector of images
    images.resize(count, std::vector<float>(rows * cols, 0.0f));

    // Read each image
    for (int i = 0; i < count; ++i) {
        for (int r = 0; r < rows * cols; ++r) {
            unsigned char pixel = 0;
            file.read(reinterpret_cast<char *>(&pixel), sizeof(pixel));
            // Normalize to [-1, 1]
            images[i][r] = (pixel / 255.0f) * 2.0f - 1.0f;
        }
    }

    file.close();
    return true;
}

// Reads MNIST labels from the IDX file
bool readMNISTLabels(const std::string &filename, std::vector<int> &labels, int count) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filename << std::endl;
        return false;
    }

    int magicNumber = 0;
    int numberOfLabels = 0;

    file.read(reinterpret_cast<char *>(&magicNumber), sizeof(magicNumber));
    file.read(reinterpret_cast<char *>(&numberOfLabels), sizeof(numberOfLabels));
    magicNumber = reverseInt(magicNumber);
    numberOfLabels = reverseInt(numberOfLabels);

    // Safety check
    if (numberOfLabels < count) {
        std::cerr << "File contains fewer labels (" << numberOfLabels << ") than required ("
                << count << "). Using only " << numberOfLabels << " labels." << std::endl;
        count = numberOfLabels;
    }

    labels.resize(count);
    for (int i = 0; i < count; ++i) {
        unsigned char label = 0;
        file.read(reinterpret_cast<char *>(&label), 1);
        labels[i] = static_cast<int>(label);
    }

    file.close();
    return true;
}

// --------------------- Activation (Scaled Tanh) ---------------------------- //
inline float scaledTanh(float x) {
    // y = 1.7159 * tanh( (2/3)*x )
    const float alpha = 1.7159f;
    const float beta = 2.0f / 3.0f;
    return alpha * std::tanh(beta * x);
}

inline float scaledTanhDerivative(float x) {
    // derivative of 1.7159 * tanh( (2/3)*x )
    // = 1.7159 * (2/3) * sech^2( (2/3)*x )
    // We can use 1 - tanh^2(z) for sech^2(z).
    const float alpha = 1.7159f;
    const float beta = 2.0f / 3.0f;
    float th = std::tanh(beta * x);
    float sech2 = 1.0f - th * th;
    return alpha * beta * sech2;
}

// -------------------- Helper: Random Initialization ------------------------- //
inline float randWeight() {
    // Small random init in [-0.05, 0.05]
    return ((std::rand() / static_cast<float>(RAND_MAX)) * 0.1f) - 0.05f;
}

// -------------------- MLP Class Definition --------------------------------- //
class MLP {
public:
    MLP() {
        // Allocate weight/bias for layers:
        // Layer 1: Input -> Hidden1
        W1.resize(INPUT_SIZE, std::vector<float>(HIDDEN1));
        b1.resize(HIDDEN1);

        // Layer 2: Hidden1 -> Hidden2
        W2.resize(HIDDEN1, std::vector<float>(HIDDEN2));
        b2.resize(HIDDEN2);

        // Layer 3: Hidden2 -> Output
        W3.resize(HIDDEN2, std::vector<float>(NUM_CLASSES));
        b3.resize(NUM_CLASSES);

        // Random init
        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < HIDDEN1; ++j) {
                W1[i][j] = randWeight();
            }
        }
        for (int j = 0; j < HIDDEN1; ++j) {
            b1[j] = randWeight();
        }

        for (int i = 0; i < HIDDEN1; ++i) {
            for (int j = 0; j < HIDDEN2; ++j) {
                W2[i][j] = randWeight();
            }
        }
        for (int j = 0; j < HIDDEN2; ++j) {
            b2[j] = randWeight();
        }

        for (int i = 0; i < HIDDEN2; ++i) {
            for (int j = 0; j < NUM_CLASSES; ++j) {
                W3[i][j] = randWeight();
            }
        }
        for (int j = 0; j < NUM_CLASSES; ++j) {
            b3[j] = randWeight();
        }
    }

    // Forward pass for a single sample
    void forward(const std::vector<float> &x) {
        // 1) Input -> Hidden1
        z1.resize(HIDDEN1);
        a1.resize(HIDDEN1);
        for (int j = 0; j < HIDDEN1; ++j) {
            float sum = b1[j];
            for (int i = 0; i < INPUT_SIZE; ++i) {
                sum += x[i] * W1[i][j];
            }
            z1[j] = sum;
            a1[j] = scaledTanh(sum);
        }

        // 2) Hidden1 -> Hidden2
        z2.resize(HIDDEN2);
        a2.resize(HIDDEN2);
        for (int j = 0; j < HIDDEN2; ++j) {
            float sum = b2[j];
            for (int i = 0; i < HIDDEN1; ++i) {
                sum += a1[i] * W2[i][j];
            }
            z2[j] = sum;
            a2[j] = scaledTanh(sum);
        }

        // 3) Hidden2 -> Output (logits)
        z3.resize(NUM_CLASSES);
        a3.resize(NUM_CLASSES); // final output
        for (int j = 0; j < NUM_CLASSES; ++j) {
            float sum = b3[j];
            for (int i = 0; i < HIDDEN2; ++i) {
                sum += a2[i] * W3[i][j];
            }
            z3[j] = sum;
        }

        // Here we could do a scaledTanh on the output, but for classification,
        // let's apply a "softmax" for the final class probabilities.
        // However, the paper used a scaled tanh for output as well, or MSE for digits.
        // Let's do a simple softmax for classification. You can also do scaled tanh if you prefer.
        float maxLogit = *std::max_element(z3.begin(), z3.end());
        float sumExp = 0.0f;
        for (int j = 0; j < NUM_CLASSES; ++j) {
            float e = std::exp(z3[j] - maxLogit);
            a3[j] = e;
            sumExp += e;
        }
        for (int j = 0; j < NUM_CLASSES; ++j) {
            a3[j] /= sumExp; // final probability
        }
    }

    // Backprop for a single sample (cross-entropy loss with softmax)
    void backward(const std::vector<float> &x, int label) {
        // 1) Compute output layer delta
        // If using cross-entropy + softmax, derivative is (p_j - y_j)
        std::vector<float> delta3(NUM_CLASSES);
        for (int j = 0; j < NUM_CLASSES; ++j) {
            float target = (j == label) ? 1.0f : 0.0f;
            delta3[j] = (a3[j] - target);
        }

        // 2) Hidden2 -> Output weight/bias update
        for (int j = 0; j < NUM_CLASSES; ++j) {
            for (int i = 0; i < HIDDEN2; ++i) {
                W3[i][j] -= LEARNING_RATE * delta3[j] * a2[i];
            }
            b3[j] -= LEARNING_RATE * delta3[j];
        }

        // 3) Delta for hidden2
        std::vector<float> delta2(HIDDEN2, 0.0f);
        for (int i = 0; i < HIDDEN2; ++i) {
            float grad = 0.0f;
            for (int j = 0; j < NUM_CLASSES; ++j) {
                grad += delta3[j] * W3[i][j];
            }
            // multiply by derivative of scaled tanh at z2[i]
            grad *= scaledTanhDerivative(z2[i]);
            delta2[i] = grad;
        }

        // 4) Hidden1 -> Hidden2 weight/bias update
        for (int j = 0; j < HIDDEN2; ++j) {
            for (int i = 0; i < HIDDEN1; ++i) {
                W2[i][j] -= LEARNING_RATE * delta2[j] * a1[i];
            }
            b2[j] -= LEARNING_RATE * delta2[j];
        }

        // 5) Delta for hidden1
        std::vector<float> delta1(HIDDEN1, 0.0f);
        for (int i = 0; i < HIDDEN1; ++i) {
            float grad = 0.0f;
            for (int j = 0; j < HIDDEN2; ++j) {
                grad += delta2[j] * W2[i][j];
            }
            grad *= scaledTanhDerivative(z1[i]);
            delta1[i] = grad;
        }

        // 6) Input -> Hidden1 weight/bias update
        for (int j = 0; j < HIDDEN1; ++j) {
            for (int i = 0; i < INPUT_SIZE; ++i) {
                W1[i][j] -= LEARNING_RATE * delta1[j] * x[i];
            }
            b1[j] -= LEARNING_RATE * delta1[j];
        }
    }

    // Predict label given an input sample
    int predict(const std::vector<float> &x) {
        forward(x);
        // pick max prob
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
    // Weights & biases
    std::vector<std::vector<float> > W1; // [INPUT_SIZE][HIDDEN1]
    std::vector<float> b1; // [HIDDEN1]

    std::vector<std::vector<float> > W2; // [HIDDEN1][HIDDEN2]
    std::vector<float> b2; // [HIDDEN2]

    std::vector<std::vector<float> > W3; // [HIDDEN2][NUM_CLASSES]
    std::vector<float> b3; // [NUM_CLASSES]

    // Intermediate forward results
    std::vector<float> z1, a1;
    std::vector<float> z2, a2;
    std::vector<float> z3, a3;
};

// --------------------------- Main Program ----------------------------------- //
int main(int argc, char **argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                << " <train_images> <train_labels> <test_images> <test_labels>\n";
        return 1;
    }

    std::string trainImagePath = argv[1];
    std::string trainLabelPath = argv[2];
    std::string testImagePath = argv[3];
    std::string testLabelPath = argv[4];

    // Seed random
    std::srand(123); // fix seed for reproducibility (optional)

    // Read training data
    std::vector<std::vector<float> > trainImages;
    std::vector<int> trainLabels;
    if (!readMNISTImages(trainImagePath, trainImages, TRAINING_SAMPLES) ||
        !readMNISTLabels(trainLabelPath, trainLabels, TRAINING_SAMPLES)) {
        std::cerr << "Error reading training data.\n";
        return 1;
    }
    // Read test data
    std::vector<std::vector<float> > testImages;
    std::vector<int> testLabels;
    if (!readMNISTImages(testImagePath, testImages, TEST_SAMPLES) ||
        !readMNISTLabels(testLabelPath, testLabels, TEST_SAMPLES)) {
        std::cerr << "Error reading test data.\n";
        return 1;
    }

    // Create MLP
    MLP dnn;

    // Simple training loop
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        // Shuffle training data indices if you like
        // (Here, we skip shuffling for brevity; you can implement it for better generalization)

        float epochLoss = 0.0f;
        int correctCount = 0;

        for (int i = 0; i < TRAINING_SAMPLES; ++i) {
            // Forward
            dnn.forward(trainImages[i]);

            // Compute cross-entropy loss for logging
            float logProb = dnn.predict(trainImages[i]) == trainLabels[i] ? 0.0f : 0.0f;
            // (We do partial demonstration; you can refine if you want to compute real CE.)

            // Check if predicted correct
            int pred = dnn.predict(trainImages[i]);
            if (pred == trainLabels[i]) {
                correctCount++;
            }

            // Backprop
            dnn.backward(trainImages[i], trainLabels[i]);

            // Accumulate a dummy loss measure
            // In practice, you'd want the actual cross-entropy, e.g. -log(a3[label]).
            epochLoss += (pred == trainLabels[i]) ? 0.0f : 1.0f;
        }

        float trainAccuracy = (100.0f * correctCount) / TRAINING_SAMPLES;
        std::cout << "Epoch " << (epoch + 1)
                << " - Training Accuracy: " << trainAccuracy
                << "%, Loss (not real CE): " << epochLoss
                << std::endl;

        // Evaluate on test set (optional each epoch; can be time-consuming)
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

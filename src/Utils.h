#ifndef UTILS_H
#define UTILS_H

#include <math.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace Utils {

/**
 * @brief Generates a random number between 0 and 1.
 *
 * @return double The random number.
 */
static double getRandom() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<double> dis(0.0, 1.0);
    return dis(gen);
}

/**
 * @brief Normalizes a vector of data to have a mean of 0 and standard deviation of 1.
 *
 * @param data The input data vector.
 * @return std::vector<double> The normalized data vector.
 */
static std::vector<double> normalize(const std::vector<double>& data) {
    double mean = 0.0, sd = 0.0;

    // Calculate the mean
    for (auto& d : data) {
        mean += d;
    }
    mean /= data.size();

    // Calculate the standard deviation
    for (auto& d : data) {
        sd += std::pow(d - mean, 2);
    }
    sd = std::sqrt(sd / (data.size() - 1));

    // Normalize the data
    std::vector<double> normalizedData(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        normalizedData[i] = (data[i] - mean) / sd;
    }

    return normalizedData;
}

/**
 * @brief Converts a 28x28 image into a vector of size 784.
 *
 * @param image The input image vector.
 * @return std::vector<double> The vector representation of the image.
 */
static std::vector<double> toVector(const std::vector<uint8_t>& image) {
    std::vector<double> vec(784);
    for (size_t i = 0; i < 784; i++) {
        vec[i] = static_cast<double>(image[i]) / 255.0;
    }
    return vec;
}

/**
 * @brief Returns the index of the maximum value in a vector.
 *
 * @param output The input vector.
 * @return size_t The index of the maximum value.
 */
static size_t maxIndex(const std::vector<double>& output) {
    double maxVal = output[0];
    size_t maxIndex = 0;
    for (size_t i = 1; i < output.size(); i++) {
        if (output[i] > maxVal) {
            maxVal = output[i];
            maxIndex = i;
        }
    }
    return maxIndex;
}

/**
 * @brief Calculates the mean squared error (MSE) between two vectors.
 *
 * @param expected The vector of expected values.
 * @param actual The vector of actual values.
 * @return double The mean squared error.
 * @throws std::invalid_argument If the expected and actual vectors have different sizes.
 */
static double mse(std::vector<double> expected, std::vector<double> actual) {
    if (expected.size() != actual.size()) {
        throw std::invalid_argument("Expected and actual vectors must have the same size.");
    }
    double error_sum = 0.0;
    for (int i = 0; i < expected.size(); i++) {
        double error = expected[i] - actual[i];
        error_sum += error * error;
    }
    double mse = error_sum / expected.size();
    return mse;
}

/**
 * @brief Splits the input and output data into training and validation sets.
 *
 * @param inputs The input data.
 * @param outputs The output data.
 * @param validationFraction The fraction of examples to use for validation.
 * @param trainInputs The vector to store the training input data.
 * @param trainOutputs The vector to store the training output data.
 * @param valInputs The vector to store the validation input data.
 * @param valOutputs The vector to store the validation output data.
 */
static void splitData(const std::vector<std::vector<double>>& inputs,
                      const std::vector<std::vector<double>>& outputs, double validationFraction,
                      std::vector<std::vector<double>>& trainInputs,
                      std::vector<std::vector<double>>& trainOutputs,
                      std::vector<std::vector<double>>& valInputs,
                      std::vector<std::vector<double>>& valOutputs) {
    // Determine the number of examples for validation
    size_t numValExamples = static_cast<size_t>(inputs.size() * validationFraction);

    // Select random indices for validation examples
    std::vector<size_t> valIndices;
    std::unordered_set<size_t> valIndicesSet;
    while (valIndicesSet.size() < numValExamples) {
        size_t index = rand() % inputs.size();
        if (valIndicesSet.find(index) == valIndicesSet.end()) {
            valIndices.push_back(index);
            valIndicesSet.insert(index);
        }
    }

    // Separate the training and validation examples
    trainInputs.reserve(inputs.size() - numValExamples);
    trainOutputs.reserve(outputs.size() - numValExamples);
    valInputs.reserve(numValExamples);
    valOutputs.reserve(numValExamples);
    for (size_t i = 0; i < inputs.size(); i++) {
        if (valIndicesSet.find(i) != valIndicesSet.end()) {
            valInputs.push_back(inputs[i]);
            valOutputs.push_back(outputs[i]);
        } else {
            trainInputs.push_back(inputs[i]);
            trainOutputs.push_back(outputs[i]);
        }
    }
}

/**
 * @brief Calculates the accuracy of a network on a given dataset.
 *
 * @param nn The neural network.
 * @param inputs The input data.
 * @param expectedOutputs The expected output data.
 * @return double The accuracy of the network.
 */
static double testAccuracy(Network& nn, const std::vector<std::vector<double>>& inputs,
                           const std::vector<std::vector<double>>& expectedOutputs) {
    unsigned numCorrect = 0;
    for (size_t i = 0; i < inputs.size(); i++) {
        std::vector<double> output = nn.predict(inputs[i]);
        int predictedLabel = Utils::maxIndex(output);
        int trueLabel = Utils::maxIndex(expectedOutputs[i]);
        if (predictedLabel == trueLabel) {
            numCorrect++;
        }
    }
    return (double)numCorrect / inputs.size();
}

}  // namespace Utils

#endif  // UTILS_H

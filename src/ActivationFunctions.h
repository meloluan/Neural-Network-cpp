#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace Functions {
// Activation functions and their derivatives

/**
 * @brief Calculates the sigmoid activation function for the given input.
 *
 * @param x The input value.
 * @return double The output of the sigmoid activation function.
 */
static double sigmoid(double x) {
    auto ret = 1.0 / (1.0 + std::exp(-x));

    if (std::isnan(ret)) {
        std::cout << "x = " << x << "\n";
        // throw std::runtime_error("NaN value in sigmoid");
    }
    return ret;
}

/**
 * @brief Calculates the derivative of the sigmoid activation function for the given input.
 *
 * @param x The input value.
 * @return double The derivative of the sigmoid activation function.
 */
static double sigmoidDerivative(double x) {
    double sig = sigmoid(x);
    auto ret = sig * (1 - sig);

    if (std::isnan(ret)) {
        throw std::runtime_error("NaN value in sigmoid derivative");
    }

    return ret;
}

/**
 * @brief Calculates the ReLU activation function for the given input.
 *
 * @param x The input value.
 * @return double The output of the ReLU activation function.
 */
static double reLU(double x) {
    return std::max(0.0, x);
}

/**
 * @brief Calculates the derivative of the ReLU activation function for the given input.
 *
 * @param x The input value.
 * @return double The derivative of the ReLU activation function.
 */
static double reLUDerivative(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

/**
 * @brief Calculates the softmax activation function for the given inputs.
 *
 * @param inputs The input values.
 * @return std::vector<double> The outputs of the softmax activation function.
 */
static std::vector<double> softmax(const std::vector<double>& inputs) {
    std::vector<double> outputs;
    double maxInput = *std::max_element(inputs.begin(), inputs.end());

    double sum = 0.0;
    for (double input : inputs) {
        double output = std::exp(input - maxInput);
        sum += output;
        outputs.push_back(output);
    }

    for (double& output : outputs) {
        output /= sum;
        if (std::isnan(output)) {
            throw std::runtime_error("NaN value in softmax");
        }
    }

    return outputs;
}

/**
 * @brief Calculates the derivative of the softmax activation function for the given inputs.
 *
 * @param x The input values.
 * @return std::vector<double> The derivatives of the softmax activation function.
 */
static std::vector<double> softmaxDerivative(const std::vector<double>& x) {
    std::vector<double> softm = softmax(x);
    std::vector<double> derivative(x.size());

    for (size_t i = 0; i < x.size(); i++) {
        derivative[i] = softm[i] * (1 - softm[i]);

        if (std::isnan(derivative[i])) {
            throw std::runtime_error("NaN value in softmax derivative");
        }
    }

    return derivative;
}

}  // namespace Functions

#endif  // ACTIVATION_FUNCTIONS_H

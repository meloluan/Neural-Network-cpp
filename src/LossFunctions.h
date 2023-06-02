#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

#include <cmath>
#include <vector>

#include "Layer.h"
#include "Network.h"

namespace LossFunctions {

/**
 * @brief Calculates the mean squared error loss between the expected outputs and the forward
 * output.
 *
 * @param expectedOutputs The expected output values.
 * @param forwardOutput The forward output values.
 * @return double The mean squared error loss.
 */
static double meanSquaredErrorLoss(const std::vector<double>& expectedOutputs,
                                   const std::vector<double>& forwardOutput) {
    double loss = 0.0;

    // Calculate error
    double sampleLoss = 0.0;
    for (size_t j = 0; j < expectedOutputs.size(); j++) {
        double diff = expectedOutputs[j] - forwardOutput[j];
        sampleLoss += diff * diff;
    }
    sampleLoss /= expectedOutputs.size();

    loss += sampleLoss;

    return loss;
}

/**
 * @brief Calculates the cross-entropy loss between the expected outputs and the forward output.
 *
 * @param expectedOutputs The expected output values.
 * @param forwardOutput The forward output values.
 * @return double The cross-entropy loss.
 */
static double crossEntropyLoss(const std::vector<double>& expectedOutputs,
                               const std::vector<double>& forwardOutput) {
    // Calculate cross-entropy loss
    double loss = 0.0;
    for (size_t j = 0; j < expectedOutputs.size(); j++) {
        double output = forwardOutput[j];
        if (output < std::numeric_limits<double>::epsilon()) {
            output = std::numeric_limits<double>::epsilon();
        } else if (output > 1.0 - std::numeric_limits<double>::epsilon()) {
            output = 1.0 - std::numeric_limits<double>::epsilon();
        }
        loss -= expectedOutputs[j] * std::log(output);
    }
    return loss;
}

/**
 * @brief Calculates the binary cross-entropy loss between the expected outputs and the forward
 * output.
 *
 * @param expectedOutputs The expected output values.
 * @param forwardOutput The forward output values.
 * @return double The binary cross-entropy loss.
 */
static double binaryCrossEntropyLoss(const std::vector<double>& expectedOutputs,
                                     const std::vector<double>& forwardOutput) {
    // Calculate binary cross-entropy loss
    double loss = 0.0;
    for (size_t j = 0; j < expectedOutputs.size(); j++) {
        double output = forwardOutput[j];
        if (output < std::numeric_limits<double>::epsilon()) {
            output = std::numeric_limits<double>::epsilon();
        } else if (output > 1.0 - std::numeric_limits<double>::epsilon()) {
            output = 1.0 - std::numeric_limits<double>::epsilon();
        }
        loss -= expectedOutputs[j] * std::log(output) +
                (1.0 - expectedOutputs[j]) * std::log(1.0 - output);
    }
    return loss;
}

}  // namespace LossFunctions

#endif  // LOSS_FUNCTIONS_H

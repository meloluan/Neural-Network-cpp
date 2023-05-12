#include <cmath>
#include <vector>

#include "Layer.h"
#include "Network.h"

namespace LossFunctions {

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

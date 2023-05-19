#ifndef LAYER_H
#define LAYER_H

#include <functional>
#include <vector>

#include "ActivationFunctions.h"
#include "Neuron.h"

enum class ActivationFunctionType { Sigmoid, ReLU, Softmax };

enum class LossFunctionType { None, MeanSquaredError, CrossEntropy, BinaryCrossEntropy };

class Layer {
public:
    using Activation = std::function<double(double)>;
    using LossFunction =
        std::function<double(const std::vector<double>&, const std::vector<double>&)>;

    Layer(unsigned int inputSize, unsigned int outputSize,
          ActivationFunctionType activationFunction,
          LossFunctionType lossFunction = LossFunctionType::None);

    std::vector<double> feedForward(const std::vector<double>& inputs);

    void adjustWeights(const std::vector<double>& inputs, const std::vector<double>& deltas,
                       double learningRate);

    double activationFunction(double input);
    double activationDerivativeFunction(double input);

    std::vector<Neuron> neurons;

    std::vector<double> getOutputs();

    LossFunction getLossFunction();

    size_t getOutputSize();
    size_t getInputSize();

    std::vector<std::vector<double>> getWeights();

    bool withSoftmax();

private:
    Activation activation;
    Activation activationDerivative;

    LossFunctionType m_lossFunction = LossFunctionType::None;

    bool m_withSoftmax = false;

    std::pair<Activation, Activation> getActivationFunctions(
        ActivationFunctionType activationFunction);

    std::vector<double> m_outputs;
};

#endif  // LAYER_H

#include "Layer.h"

#include "LossFunctions.h"

Layer::Layer(unsigned int inputSize, unsigned int outputSize,
             ActivationFunctionType activationFunction, LossFunctionType lossFunction)
        : m_lossFunction(lossFunction) {
    m_outputs.resize(outputSize);

    for (size_t i = 0; i < outputSize; i++) {
        m_outputs[i] = 0.0;
        neurons.emplace_back(inputSize);
    }

    if (activationFunction == ActivationFunctionType::Softmax) {
        m_withSoftmax = true;
    } else {
        std::tie(activation, activationDerivative) = getActivationFunctions(activationFunction);
    }
}

Layer::LossFunction Layer::getLossFunction() {
    switch (m_lossFunction) {
    case LossFunctionType::CrossEntropy:
        return LossFunctions::crossEntropyLoss;
        break;
    case LossFunctionType::MeanSquaredError:
        return LossFunctions::meanSquaredErrorLoss;
        break;
    case LossFunctionType::BinaryCrossEntropy:
        return LossFunctions::binaryCrossEntropyLoss;
        break;
    default:
        break;
    }
}

std::vector<double> Layer::feedForward(const std::vector<double>& inputs) {
    m_inputValues = inputs;  // save the inputs
    for (size_t i = 0; i < neurons.size(); i++) {
        m_outputs[i] = neurons[i].feedForward(inputs);
    }

    if (m_withSoftmax) {
        m_outputs = Functions::softmax(m_outputs);
    } else {
        for (size_t i = 0; i < m_outputs.size(); i++) {
            m_outputs[i] = activation(m_outputs[i]);
        }
    }

    return m_outputs;
}

std::vector<double> Layer::getOutputs() {
    return m_outputs;
}

std::vector<double> Layer::getInputValues() const {
    return m_inputValues;
}

std::vector<std::vector<double>> Layer::getWeights() {
    std::vector<std::vector<double>> weights(neurons.size());
    for (size_t i = 0; i < neurons.size(); i++) {
        weights[i] = neurons[i].getWeights();
    }
    return weights;
}

void Layer::adjustWeights(const std::vector<double>& inputs, const std::vector<double>& deltas,
                          double learningRate, double regularizationTerm) {
    for (size_t i = 0; i < neurons.size(); i++) {
        neurons[i].adjustWeights(inputs, deltas[i], learningRate, regularizationTerm);
    }
}

bool Layer::withSoftmax() {
    return m_withSoftmax;
}

std::pair<Layer::Activation, Layer::Activation> Layer::getActivationFunctions(
    ActivationFunctionType activationFunction) {
    switch (activationFunction) {
    case ActivationFunctionType::Sigmoid:
        return std::make_pair(Functions::sigmoid, Functions::sigmoidDerivative);
    case ActivationFunctionType::ReLU:
        return std::make_pair(Functions::reLU, Functions::reLUDerivative);
    case ActivationFunctionType::Softmax:
        throw std::runtime_error(
            "Softmax activation function is not supported for individual neurons. Use it only for "
            "the output layer.");
    default:
        throw std::runtime_error("Unsupported activation function");
    }
}

double Layer::activationFunction(double input) {
    return activation(input);
}

double Layer::activationDerivativeFunction(double input) {
    return activationDerivative(input);
}

size_t Layer::getOutputSize() {
    return neurons.size();
}

size_t Layer::getInputSize() {
    return neurons[0].getWeights().size();
}

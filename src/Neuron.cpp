#include "Neuron.h"

#include <sstream>
Neuron::Neuron(unsigned numInputs) : bias(0.0) {
    // Inicializa os pesos e o bias com valores aleatórios
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0.0, 1.0);

    weights.resize(numInputs);

    for (double& w : weights) {
        w = d(gen);
    }
    bias = d(gen);
}

double Neuron::feedForward(const std::vector<double>& inputs) const {
    double sum = 0.0;

    for (size_t i = 0; i < inputs.size(); i++) {
        sum += inputs[i] * weights[i];
    }

    sum += bias;
    return sum;
}

void Neuron::adjustWeights(const std::vector<double>& inputs, double delta, double learningRate) {
    if (inputs.size() != weights.size()) {
        std::stringstream ss;
        ss << "Mismatch in the number of inputs and weights: " << inputs.size()
           << " != " << weights.size();
        throw std::runtime_error(ss.str());
    }

    for (size_t i = 0; i < inputs.size(); i++) {
        weights[i] += learningRate * delta * inputs[i];
    }

    bias += learningRate * delta;
}

void Neuron::setBias(double bias) {
    this->bias = bias;
}

#include "Neuron.h"

#include <sstream>
Neuron::Neuron(unsigned numInputs) : m_bias(0.0) {
    // Inicializa os pesos e o bias com valores aleatórios
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0.0, sqrt(2.0 / numInputs));  // He Initialization

    // std::normal_distribution<> d(0, 1);  // Normal Distribution

    m_weights.resize(numInputs);

    for (double& w : m_weights) {
        w = d(gen);
    }
    m_bias = d(gen);
}

double Neuron::feedForward(const std::vector<double>& inputs) const {
    double sum = 0.0;

    for (size_t i = 0; i < inputs.size(); i++) {
        sum += inputs[i] * m_weights[i];
    }

    sum += m_bias;
    return sum;
}

void Neuron::adjustWeights(const std::vector<double>& inputs, double delta, double learningRate,
                           double regularizationTerm) {
    if (inputs.size() != m_weights.size()) {
        std::stringstream ss;
        ss << "Mismatch in the number of inputs and weights: " << inputs.size()
           << " != " << m_weights.size();
        throw std::runtime_error(ss.str());
    }

    if (regularizationTerm == 0.0) {  // No regularization
        for (size_t i = 0; i < inputs.size(); i++) {
            m_weights[i] += learningRate * delta * inputs[i];
        }

    } else {  // L2 regularization
        for (size_t j = 0; j < m_weights.size(); j++) {
            double regularization = regularizationTerm * m_weights[j];
            double deltaWeight = learningRate * (delta * inputs[j] - regularization);
            m_weights[j] += deltaWeight;
        }
    }

    m_bias += learningRate * delta;
}

void Neuron::setBias(double bias) {
    m_bias = bias;
}

double Neuron::getWeight(unsigned index) const {
    return m_weights[index];
}

std::vector<double> Neuron::getWeights() const {
    return m_weights;
}

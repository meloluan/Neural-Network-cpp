#include "Neuron.h"

#include <iostream>

#include "Utils.h"

// set the number of inputs and initialize the weights and bias randomly
void Neuron::initialize(int num_inputs) {
    // Initialize the weights and bias randomly
    m_weights.resize(num_inputs);
    m_inputs.resize(num_inputs);
    for (int i = 0; i < num_inputs; i++) {
        m_weights[i] = Utils::getRandom();
    }
    m_bias = Utils::getRandom();
}

// calculate the output of the neuron given the input values
double Neuron::calculateOutput(std::vector<double> inputs) {
    if (inputs.size() > m_inputs.size()) {
        std::cout << "ERROR: Number of inputs is greater than the number of weights" << std::endl;
    }
    // Calculate the weighted sum of inputs and add the bias
    double net = m_bias;
    for (int i = 0; i < inputs.size(); i++) {
        net += inputs[i] * m_weights[i];
    }

    // Store the inputs for later use
    m_inputs = inputs;

    // Apply the activation function (sigmoid)
    m_output = Utils::sigmoid(net);
    return m_output;
}

// update the weights and bias of the neuron during backpropagation
void Neuron::updateWeights(double learningRate, double error) {
    // Calculate the delta value for the neuron
    m_delta = error * Utils::sigmoidDerivative(m_output);

    // Update the weights and bias of the neuron
    for (int i = 0; i < m_weights.size(); i++) {
        m_weights[i] += learningRate * m_delta * m_inputs[i];
    }
    m_bias += learningRate * m_delta;
}

// getters for the output and delta values
double Neuron::getOutput() const {
    return m_output;
}

double Neuron::getDelta() const {
    return m_delta;
}

void Neuron::setDelta(double delta) {
    m_delta = delta;
}

std::vector<double> Neuron::getWeights() const {
    return m_weights;
}

void Neuron::setWeight(int index, double weight) {
    m_weights[index] = weight;
}

std::vector<double> Neuron::getInputs() const {
    return m_inputs;
}

void Neuron::setInputs(std::vector<double> inputs) {
    m_inputs = inputs;
}

int Neuron::getNumInputs() const {
    return m_inputs.size();
}

double Neuron::getWeight(int index) const {
    return m_weights[index];
}

double Neuron::getActivationDerivative() const {
    return Utils::sigmoidDerivative(m_output);
}

double Neuron::getBias() const {
    return m_bias;
}

void Neuron::setBias(double bias) {
    m_bias = bias;
}
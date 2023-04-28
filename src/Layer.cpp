#include "Layer.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "Utils.h"

// add a neuron to the layer
void Layer::addNeuron(Neuron neuron) {
    m_neurons.push_back(neuron);
}

// set the number of inputs and initialize the m_neurons in the layer
void Layer::initialize(int numInputs, int numNeurons, bool isOutputLayer) {
    m_isOutputLayer = isOutputLayer;
    m_numInputs = numInputs;
    m_numNeurons = numNeurons;

    std::cout << "Initializing layer with " << m_numNeurons << " m_neurons and " << numInputs
              << " inputs" << std::endl;

    m_neurons.resize(m_numNeurons);
    for (auto& neuron : m_neurons) {
        neuron.initialize(numInputs);
    }

    m_outputs.resize(m_numNeurons);
}

// calculate the output of each neuron in the layer given the input values
std::vector<double> Layer::forwardPropagate(std::vector<double> inputs) {
    for (int i = 0; i < m_neurons.size(); i++) {
        m_outputs[i] = m_neurons[i].calculateOutput(inputs);
    }

    // if (m_isOutputLayer) {
    //     m_outputs = Utils::sigmoid(m_outputs);
    // }
    return m_outputs;
}

// backpropagate the error through the layer
void Layer::backpropagate(const std::vector<double>& frontLayerErrors, double learningRate) {
    if (m_isOutputLayer) {
        backpropagateOutputLayer(frontLayerErrors);
    } else {
        backpropagateHiddenLayer(frontLayerErrors, learningRate);
    }
}

void Layer::backpropagateOutputLayer(const std::vector<double>& errors) {
    for (auto i = 0u; i < m_neurons.size(); ++i) {
        auto& neuron = m_neurons[i];
        auto output = neuron.getOutput();
        auto error = errors[i];

        auto delta = error * Utils::sigmoidDerivative(output);
        neuron.setDelta(delta);
    }
}

void Layer::backpropagateHiddenLayer(const std::vector<double>& frontLayerDelta,
                                     double learningRate) {
    for (auto i = 0u; i < m_neurons.size(); ++i) {
        auto& neuron = m_neurons[i];
        auto output = neuron.getOutput();
        auto weights = neuron.getWeights();
        auto error = 0.0;

        for (auto j = 0u; j < frontLayerDelta.size(); j++) {
            error += frontLayerDelta[j] * weights[j];
        }
        neuron.updateWeights(learningRate, error);
    }
}

// update the weights and biases of each neuron in the layer during backpropagation
void Layer::updateWeights(double learningRate) {
    for (auto& neuron : m_neurons) {
        const auto& inputs = neuron.getInputs();
        const auto& weights = neuron.getWeights();
        double delta = neuron.getDelta();

        for (auto i = 0u; i < inputs.size(); ++i) {
            double weightDelta = learningRate * delta * inputs[i];
            double weight = weights[i];
            double newWeight = weight + weightDelta;
            neuron.setWeight(i, newWeight);
        }

        double biasDelta = learningRate * delta;
        double bias = neuron.getBias();
        double newBias = bias + biasDelta;
        neuron.setBias(newBias);
    }
}
// get the number of m_neurons in the layer
int Layer::getNumNeurons() const {
    return m_neurons.size();
}

// get the output values of each neuron in the layer
std::vector<double> Layer::getOutputs() const {
    std::vector<double> outputs(m_neurons.size());
    for (auto i = 0u; i < m_neurons.size(); ++i) {
        outputs[i] = m_neurons[i].getOutput();
    }
    return outputs;
}

double Layer::getOutput(int index) const {
    return m_neurons[index].getOutput();
}

// get the deltas of each neuron in the layer
std::vector<double> Layer::getDeltas() const {
    std::vector<double> deltas(m_neurons.size());
    for (int i = 0; i < m_neurons.size(); i++) {
        deltas[i] = m_neurons[i].getDelta();
    }
    return deltas;
}

std::vector<double> Layer::getWeights() const {
    std::vector<double> weights;
    for (int i = 0; i < m_neurons.size(); i++) {
        auto neuronWeights = m_neurons[i].getWeights();
        for (int j = 0; j < neuronWeights.size(); j++) {
            weights.push_back(neuronWeights[j]);
        }
    }
    return weights;
}

int Layer::getNumInputs() const {
    return m_numInputs;
}

std::vector<double> Layer::getInputs() const {
    std::vector<double> inputs;
    for (int i = 0; i < m_neurons.size(); i++) {
        auto neuronInputs = m_neurons[i].getInputs();
        for (int j = 0; j < neuronInputs.size(); j++) {
            inputs.push_back(neuronInputs[j]);
        }
    }
    return inputs;
}

void Layer::setDelta(std::vector<double> deltas) {
    for (int i = 0; i < m_neurons.size(); i++) {
        m_neurons[i].setDelta(deltas[i]);
    }
}

void Layer::setDelta(int index, double delta) {
    m_neurons[index].setDelta(delta);
}

void Layer::setWeight(int index, double weight) {
    int neuronIndex = index / m_numInputs;
    int weightIndex = index % m_numInputs;
    m_neurons[neuronIndex].setWeight(weightIndex, weight);
}

void Layer::setWeight(int index, int weightIndex, double weight) {
    m_neurons[index].setWeight(weightIndex, weight);
}

Neuron Layer::getNeuron(int index) const {
    return m_neurons[index];
}

double Layer::getNeuronOutput(int index) const {
    return m_neurons[index].getOutput();
}

bool Layer::isOutputLayer() const {
    return m_isOutputLayer;
}

void Layer::setErrors(const std::vector<double>& nextLayerErrors) {
    errors.clear();
    for (int i = 0; i < m_neurons.size(); i++) {
        double error = 0.0;
        for (int j = 0; j < nextLayerErrors.size(); j++) {
            error += nextLayerErrors[j] * m_neurons[i].getWeight(j);
        }
        error *= Utils::sigmoidDerivative(m_neurons[i].getOutput());
        errors.push_back(error);
        m_neurons[i].setDelta(error);
    }
}

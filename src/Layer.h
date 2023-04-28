#pragma once

#include <vector>

#include "Neuron.h"

class Layer {
private:
    std::vector<Neuron> m_neurons;
    std::vector<double> errors;
    int m_numInputs = 0;
    int m_numNeurons = 0;
    bool m_isOutputLayer = false;
    std::vector<double> m_outputs;

public:
    // add a neuron to the layer
    void addNeuron(Neuron neuron);
    Neuron getNeuron(int index) const;

    // set the number of inputs and initialize the neurons in the layer
    void initialize(int num_inputs, int num_neurons, bool isOutputLayer = false);

    // calculate the output of each neuron in the layer given the input values
    std::vector<double> forwardPropagate(std::vector<double> inputs);

    void backpropagate(const std::vector<double>& frontLayerErrors, double learningRate);

    void backpropagateOutputLayer(const std::vector<double>& frontLayerErrors, double learningRate);

    void backpropagateHiddenLayer(const std::vector<double>& frontLayerErrors, double learningRate);

    // update the weights and biases of each neuron in the layer during backpropagation
    void updateWeights(double learningRate);

    // get the number of neurons in the layer
    int getNumNeurons() const;

    // get the output values of each neuron in the layer
    std::vector<double> getOutputs() const;
    double getOutput(int index) const;

    // get the deltas of each neuron in the layer
    std::vector<double> getDeltas() const;

    std::vector<double> getWeights() const;

    int getNumInputs() const;

    std::vector<double> getInputs() const;

    void setDelta(std::vector<double> deltas);
    void setDelta(int index, double delta);

    void setWeight(int index, double weight);
    void setWeight(int index, int weightIndex, double weight);

    double getNeuronOutput(int index) const;

    bool isOutputLayer() const;

    void setErrors(const std::vector<double>& nextLayerErrors);
};

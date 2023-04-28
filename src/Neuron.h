#pragma once

#include <vector>

class Neuron {
private:
    std::vector<double> m_inputs;
    std::vector<double> m_weights;
    double m_bias;
    double m_output;
    double m_delta;

public:
    // set the number of inputs and initialize the weights and bias randomly
    void initialize(int num_inputs);

    // calculate the output of the neuron given the input values
    double calculateOutput(std::vector<double> inputs);

    // update the weights and bias of the neuron during backpropagation
    void updateWeights(double learningRate, double error);

    // getters for the output and delta values
    double getOutput() const;

    double getDelta() const;
    void setDelta(double delta);

    std::vector<double> getWeights() const;
    void setWeight(int index, double weight);

    std::vector<double> getInputs() const;
    void setInputs(std::vector<double> inputs);

    int getNumInputs() const;

    double getWeight(int index) const;
    double getActivationDerivative() const;

    double getBias() const;
    void setBias(double bias);
};
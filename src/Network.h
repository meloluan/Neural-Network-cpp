#pragma once

#include <vector>

#include "Layer.h"

class Network {
private:
    std::vector<Layer> layers;

public:
    // add a layer to the network
    void addLayer(int numNeurons, int numInputs);

    std::vector<double> predict(const std::vector<double>& input);

    // set the number of inputs and outputs and initialize the network layers
    void initialize(int num_inputs, int num_outputs, std::vector<int> num_neurons);

    // calculate the output of the network given the input values
    std::vector<double> calculateOutput(std::vector<double> inputs);

    void backpropagate(std::vector<double> errors, double learningRate);

    // train the network using backpropagation
    void train(std::vector<std::vector<double>> training_inputs,
               std::vector<std::vector<double>> training_outputs, double learning_rate,
               int num_epochs);
};
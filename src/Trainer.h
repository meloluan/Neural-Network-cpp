#pragma once

#include <string>

#include "Network.h"

class Trainer {
private:
    Network m_network;
    std::vector<std::vector<double>> m_trainingInputs;
    std::vector<std::vector<double>> m_trainingOutputs;

public:
    Trainer(Network network, std::vector<std::vector<double>> trainingInputs,
            std::vector<std::vector<double>> trainingOutputs);

    // train the network using backpropagation
    void train(double learning_rate, int num_epochs);

    // calculate the network's output given input values
    std::vector<double> predict(std::vector<double> inputs);

    // // save the network to a file
    // void save(std::string filename);

    // // load the network from a file
    // void load(std::string filename);
};

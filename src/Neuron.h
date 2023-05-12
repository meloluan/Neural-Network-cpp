
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
class Neuron {
public:
    Neuron(unsigned numInputs);

    double feedForward(const std::vector<double>& inputs) const;

    void adjustWeights(const std::vector<double>& inputs, double delta, double learningRate);

    std::vector<double> weights;

private:
    double bias;
};
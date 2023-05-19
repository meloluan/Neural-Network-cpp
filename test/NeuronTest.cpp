#include "ActivationFunctions.h"
#include "Neuron.h"
#include "gtest/gtest.h"

// Test fixture class
class NeuronTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up the neuron
        unsigned numInputs = 3;
        neuron = std::make_unique<Neuron>(numInputs);
    }

    std::unique_ptr<Neuron> neuron;
};

// Test case for feed forward
TEST_F(NeuronTest, FeedForward) {
    // Test inputs
    std::vector<double> inputs = {0.5, 0.3, 0.2};

    // Neuron weights
    std::vector<double> weights = {0.1, 0.2, 0.3};
    double bias = -0.1;

    // Create a neuron
    Neuron neuron(weights.size());
    neuron.weights = weights;
    neuron.setBias(bias);

    // Calculate the weighted sum of inputs
    double weightedSum = 0.0;
    for (size_t i = 0; i < inputs.size(); i++) {
        weightedSum += inputs[i] * weights[i];
    }
    weightedSum += bias;

    // Calculate the expected output using the sigmoid activation function
    double expectedOutput = 1.0 / (1.0 + std::exp(-weightedSum));

    // Calculate the output using feedForward
    double output = neuron.feedForward(inputs);

    // Check if the output is close to the expected output within a tolerance of 1e-6
    ASSERT_NEAR(output, expectedOutput, 1e-6);
}

// Test case for adjusting weights
TEST_F(NeuronTest, AdjustWeights) {
    std::vector<double> inputs{0.8, 0.2, 0.4};
    double delta = 0.1;
    double learningRate = 0.01;

    // Save the initial weights
    std::vector<double> initial_weights = neuron->weights;

    neuron->adjustWeights(inputs, delta, learningRate);

    // Check if the weights have been adjusted correctly
    for (size_t i = 0; i < initial_weights.size(); i++) {
        // Replace with the expected weight adjustment based on the inputs, delta, and learning rate
        double expected_weight = initial_weights[i] + learningRate * delta * inputs[i];
        ASSERT_DOUBLE_EQ(neuron->weights[i], expected_weight);
    }
}
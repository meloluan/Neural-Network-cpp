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
    std::vector<double> inputs{0.5, 0.3, 0.2};
    double expected_output = 0.6;  // Replace with the expected output for the given inputs
    double tolerance = 0.0001;     // Replace with the desired tolerance
    double output = neuron->feedForward(inputs);
    ASSERT_NEAR(output, expected_output, tolerance);
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
        double expected_weight = initial_weights[i] + learningRate * delta * inputs[i];
        ASSERT_DOUBLE_EQ(neuron->weights[i], expected_weight);
    }
}

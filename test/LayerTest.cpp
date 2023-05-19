#include "Layer.h"
#include "gtest/gtest.h"

// Test fixture class
class LayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up the layer configuration
        unsigned int inputSize = 2;
        unsigned int outputSize = 3;
        ActivationFunctionType activationFunction = ActivationFunctionType::Sigmoid;
        LossFunctionType lossFunction = LossFunctionType::MeanSquaredError;

        // Create the layer
        layer = std::make_unique<Layer>(inputSize, outputSize, activationFunction, lossFunction);
    }

    // Common test data
    std::vector<double> inputs{0.5, 0.3};

    std::unique_ptr<Layer> layer;
};

// Test case for feed forward
TEST_F(LayerTest, FeedForward) {
    std::vector<double> output = layer->feedForward(inputs);
    ASSERT_EQ(output.size(), 3);
    ASSERT_NEAR(output[0], 0.7513650695523157, 1e-6);
    ASSERT_NEAR(output[1], 0.7456094781464974, 1e-6);
    ASSERT_NEAR(output[2], 0.7548399330985988, 1e-6);
}

// Test case for adjusting weights
TEST_F(LayerTest, AdjustWeights) {
    std::vector<double> input{0.8, 0.2};
    std::vector<double> deltas{0.1, 0.2, 0.3};
    double learningRate = 0.1;

    layer->adjustWeights(input, deltas, learningRate);

    std::vector<std::vector<double>> weights = layer->getWeights();
    ASSERT_EQ(weights.size(), 3);
    ASSERT_EQ(weights[0].size(), 2);
    ASSERT_NEAR(weights[0][0], 0.10000000000000002, 1e-6);
    ASSERT_NEAR(weights[0][1], 0.30000000000000004, 1e-6);
    ASSERT_NEAR(weights[1][0], 0.20000000000000004, 1e-6);
    ASSERT_NEAR(weights[1][1], 0.4000000000000001, 1e-6);
    ASSERT_NEAR(weights[2][0], 0.30000000000000004, 1e-6);
    ASSERT_NEAR(weights[2][1], 0.5000000000000001, 1e-6);
}

// Test case for activation function
TEST_F(LayerTest, ActivationFunction) {
    double result = layer->activationFunction(0.5);
    ASSERT_NEAR(result, 0.6224593312018546, 1e-6);
}

// Test case for activation derivative function
TEST_F(LayerTest, ActivationDerivativeFunction) {
    double result = layer->activationDerivativeFunction(0.5);
    ASSERT_NEAR(result, 0.2350037122015945, 1e-6);
}

// Test case for getting outputs
TEST_F(LayerTest, GetOutputs) {
    std::vector<double> outputs = layer->getOutputs();
    ASSERT_EQ(outputs.size(), 3);
    ASSERT_NEAR(outputs[0], 0.0, 1e-6);
    ASSERT_NEAR(outputs[1], 0.0, 1e-6);
    ASSERT_NEAR(outputs[2], 0.0, 1e-6);
}

// // Test case for getting the loss function
// TEST_F(LayerTest, GetLossFunction) {
//     LossFunctionType lossFunction = layer->getLossFunction();
//     ASSERT_EQ(lossFunction, LossFunctionType::MeanSquaredError);
// }

// Test case for getting the output size
TEST_F(LayerTest, GetOutputSize) {
    size_t outputSize = layer->getOutputSize();
    ASSERT_EQ(outputSize, 3);
}

// Test case for getting the input size
TEST_F(LayerTest, GetInputSize) {
    size_t inputSize = layer->getInputSize();
    ASSERT_EQ(inputSize, 2);
}

// Test case for checking if softmax is enabled
TEST_F(LayerTest, WithSoftmax) {
    bool withSoftmax = layer->withSoftmax();
    ASSERT_EQ(withSoftmax, false);
}

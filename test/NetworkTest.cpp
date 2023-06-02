// #include "Network.h"
// #include "gtest/gtest.h"

// // Test fixture class
// class NetworkTest : public ::testing::Test {
// protected:
//     void SetUp() override {
//         // Set up the network layers and configuration
//         std::vector<std::pair<unsigned int, ActivationFunctionType>> layerConfig{
//             {2, ActivationFunctionType::Sigmoid},
//             {3, ActivationFunctionType::Sigmoid},
//             {1, ActivationFunctionType::Sigmoid}};

//         // Create the network
//         network = std::make_unique<Network>(layerConfig);
//     }

//     // Common test data
//     std::vector<std::vector<double>> inputs{{0.5, 0.3}, {0.8, 0.2}, {0.1, 0.9}};

//     std::vector<std::vector<double>> outputs{{0.2}, {0.5}, {0.8}};

//     std::unique_ptr<Network> network;
// };

// // Test case for predicting outputs
// TEST_F(NetworkTest, Predict) {
//     std::vector<double> predictedOutput = network->predict(inputs[0]);
//     ASSERT_EQ(predictedOutput.size(), 1);
//     EXPECT_NEAR(predictedOutput[0], 0.6, 0.1);
// }

// // Test case for validating network
// TEST_F(NetworkTest, Validate) {
//     double validationError = network->validate(inputs, outputs);
//     EXPECT_NEAR(validationError, 0.2, 0.1);
// }

// // Test case for calculating mean squared error
// TEST_F(NetworkTest, MeanSquaredError) {
//     double mse = network->meanSquaredError(inputs, outputs);
//     EXPECT_NEAR(mse, 0.1, 0.05);
// }

// // Test case for training the network
// TEST_F(NetworkTest, Train) {
//     unsigned numEpochs = 10;
//     double learningRate = 0.1;
//     double validationFraction = 0.2;

//     network->train(inputs, outputs, numEpochs, learningRate, validationFraction,
//                    [](unsigned epoch, double error, double validationError) {
//                        std::cout << "Epoch: " << epoch << ", Error: " << error
//                                  << ", Validation Error: " << validationError << std::endl;
//                    });
//     // Add assertions to validate the training results
// }

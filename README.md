# Neural Network Library

This is a lightweight neural network library implemented in C++ for building and training feedforward neural networks. It provides a flexible and easy-to-use interface for creating neural network models with customizable layers, activation functions, and loss functions.

## Features

- Support for multiple hidden layers with customizable activation functions
- Support for different loss functions, including mean squared error, cross-entropy, and binary cross-entropy
- Flexibility to configure the network architecture by specifying the number of neurons in each layer
- Ability to set custom weights for the network
- Support for various optimization methods, including gradient descent, momentum, and Adam

## Installation

To use this library, simply include the necessary header files in your C++ project.

## Usage

Here's an example of how to use the library to build and train a simple neural network:

```cpp
#include "Network.h"
#include "ActivationFunctions.h"
#include "LossFunctions.h"
#include "Utils.h"

int main() {
    // Define the network architecture
    std::vector<std::pair<unsigned int, ActivationFunctionType>> layerConfig = {
        {2, ActivationFunctionType::ReLU},   // Input layer with 2 neurons and ReLU activation
        {3, ActivationFunctionType::Sigmoid} // Hidden layer with 3 neurons and Sigmoid activation
        {1, ActivationFunctionType::Sigmoid} // Output layer with 1 neuron and Sigmoid activation
    };

    // Create the network
    Network network(layerConfig, LossFunctionType::MeanSquaredError);

    // Generate sample training data
    std::vector<std::vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> outputs = {{0}, {1}, {1}, {0}};

    // Train the network
    network.train(inputs, outputs, 1000, 0.1, 0.01, 0.2, [](unsigned epoch, double error, double validationError) {
        std::cout << "Epoch: " << epoch << ", Error: " << error << ", Validation Error: " << validationError << std::endl;
    });

    // Test the network
    std::vector<double> testInput = {0, 1};
    std::vector<double> prediction = network.predict(testInput);
    std::cout << "Prediction: " << prediction[0] << std::endl;

    return 0;
}
```

In this example, a neural network with an input layer, a hidden layer, and an output layer is created. The network is trained on a set of inputs and outputs using the mean squared error loss function. After training, the network is used to make predictions on a test input.

Feel free to modify the network architecture, activation functions, loss functions, and training parameters according to your specific needs.
Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.
License

This library is licensed under the MIT License. See the LICENSE file for details.

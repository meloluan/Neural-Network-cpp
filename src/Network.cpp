#include "Network.h"

#include <algorithm>
#include <iostream>

#include "Utils.h"

// add a layer to the network
void Network::addLayer(int numNeurons, int numInputs) {
    Layer layer;
    for (int i = 0; i < numNeurons; i++) {
        Neuron neuron;
        neuron.initialize(numInputs);

        layer.addNeuron(neuron);
    }
    layers.push_back(layer);
}

// Método para realizar a predição da saída da rede a partir de uma entrada
std::vector<double> Network::predict(const std::vector<double>& input) {
    // Definir a entrada da primeira camada
    std::vector<double> output = input;

    // Propagar a entrada pela rede neural
    for (int i = 0; i < layers.size(); i++) {
        output = layers[i].forwardPropagate(output);
    }

    // Retornar a saída final da rede neural
    return output;
}

// set the number of inputs and outputs and initialize the network layers
void Network::initialize(int num_inputs, int num_outputs, std::vector<int> num_neurons) {
    layers.resize(num_neurons.size());

    // initialize the input layer
    layers[0].initialize(num_inputs, num_neurons[0]);

    // initialize the hidden layers
    for (int i = 1; i < num_neurons.size() - 1; i++) {
        layers[i].initialize(num_neurons[i - 1], num_neurons[i]);
    }

    // initialize the output layer
    layers[num_neurons.size() - 1].initialize(num_neurons[num_neurons.size() - 2], num_outputs);
}

// // calculate the output of the network given the input values
std::vector<double> Network::calculateOutput(std::vector<double> inputs) {
    std::vector<double> outputs = inputs;

    // feed forward through each layer
    for (int i = 0; i < layers.size(); i++) {
        outputs = layers[i].forwardPropagate(outputs);
    }

    return outputs;
}

// update the weights and biases of each neuron in the network during backpropagation
void Network::backpropagate(std::vector<double> expectedOutput, double learningRate) {
    // calculate the output layer's errors
    auto& outputLayer = layers.back();
    auto outputLayerOutputs = outputLayer.getOutputs();
    std::vector<double> outputLayerErrors;
    for (int i = 0; i < outputLayer.getNumNeurons(); i++) {
        double error = expectedOutput[i] - outputLayerOutputs[i];
        outputLayerErrors.push_back(error);
    }

    outputLayer.backpropagate(outputLayerErrors, learningRate);

    // propagate the errors backwards through each layer
    for (int i = layers.size() - 2; i >= 0; i--) {
        auto& layer = layers[i];
        std::vector<double> layerErrors(layer.getNumNeurons());

        auto layerDeltas = layer.getDeltas();

        layer.backpropagate(layerDeltas, learningRate);
    }

    // // update the weights and biases of each neuron in each layer
    // for (int i = layers.size() - 1; i >= 0; i--) {
    //     auto& layer = layers[i];
    //     // auto layerDeltas = layer.getDeltas();
    //     layer.updateWeights(learningRate);
    // }
}

// train the network using backpropagation
void Network::train(std::vector<std::vector<double>> trainingInputs,
                    std::vector<std::vector<double>> trainingOutputs, double learningRate,
                    int numEpochs) {
    // initialize start time
    time_t start_time;
    time(&start_time);

    for (int epoch = 0; epoch < numEpochs; epoch++) {
        double error = 0.0;

        // for each training example
        for (int i = 0; i < trainingInputs.size(); i++) {
            std::vector<double> inputs = trainingInputs[i];
            std::vector<double> expectedOutputs = trainingOutputs[i];

            // feed forward to calculate the output
            std::vector<double> actualOutputs = calculateOutput(inputs);

            std::cout << "SIZE " << actualOutputs.size() << " | Actual Outputs: ";
            for (int j = 0; j < actualOutputs.size(); j++) {
                std::cout << actualOutputs[j] << " ";
            }
            std::cout << " || SIZE " << expectedOutputs.size() << " | Expected Outputs: ";
            for (int j = 0; j < expectedOutputs.size(); j++) {
                std::cout << expectedOutputs[j] << " ";
            }
            std::cout << std::endl;

            // calculate the error and backpropagate
            std::vector<double> errors(expectedOutputs.size());
            for (int j = 0; j < expectedOutputs.size(); j++) {
                errors[j] = expectedOutputs[j] - actualOutputs[j];
            }

            backpropagate(errors, learningRate);

            // accumulate the total error for this epoch
            for (int j = 0; j < errors.size(); j++) {
                error += errors[j] * errors[j];
            }
        }

        // calculate remaining training time
        time_t current_time;
        time(&current_time);
        double time_elapsed = difftime(current_time, start_time);
        double avg_time_per_epoch = time_elapsed / (epoch + 1);
        double remaining_time = avg_time_per_epoch * (numEpochs - epoch - 1);
        std::cout << "Error: " << error
                  << " | Estimated remaining training time: " << remaining_time << " seconds "
                  << std::endl;
    }
}

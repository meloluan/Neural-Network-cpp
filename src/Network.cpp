#include "Network.h"

#include <algorithm>

#include "ActivationFunctions.h"

Network::Network(std::vector<std::pair<unsigned int, ActivationFunctionType>> layerConfig,
                 LossFunctionType lossFunction) {
    auto iter = layerConfig.begin();
    unsigned prevLayerSize = iter->first;

    // Iterate through layer configurations and create layers
    for (iter; iter != layerConfig.end(); ++iter) {
        unsigned layerSize = iter->first;
        ActivationFunctionType activationFunction = iter->second;

        std::cout << "Creating layer with " << prevLayerSize << " inputs and " << layerSize
                  << " outputs" << std::endl;

        layers.emplace_back(prevLayerSize, layerSize, activationFunction, lossFunction);
        prevLayerSize = layerSize;
    }

    if (lossFunction == LossFunctionType::None) {
        throw std::invalid_argument("Loss function not specified");
    }
}

std::vector<double> Network::predict(const std::vector<double>& inputs) {
    std::vector<double> layerOutputs = inputs;

    for (Layer& layer : layers) {
        layerOutputs = layer.feedForward(layerOutputs);
    }

    return layerOutputs;
}

double Network::validate(const std::vector<std::vector<double>>& inputs,
                         const std::vector<std::vector<double>>& outputs) {
    if (inputs.size() != outputs.size()) {
        throw std::runtime_error("Mismatch in the number of input and output samples");
    }

    double totalError = 0.0;

    for (size_t i = 0; i < inputs.size(); i++) {
        std::vector<double> predicted = predict(inputs[i]);

        double error = 0.0;
        for (size_t j = 0; j < outputs[i].size(); j++) {
            double diff = outputs[i][j] - predicted[j];
            error += diff * diff;
        }
        totalError += error / outputs[i].size();
    }

    return totalError / inputs.size();
}
double Network::meanSquaredError(const std::vector<std::vector<double>>& inputs,
                                 const std::vector<std::vector<double>>& expectedOutputs) {
    if (inputs.size() != expectedOutputs.size() || inputs.empty()) {
        throw std::invalid_argument("Invalid input or output vectors");
    }
    double sumSquaredError = 0.0;
    for (size_t i = 0; i < inputs.size(); i++) {
        const std::vector<double>& input = inputs[i];
        const std::vector<double>& expectedOutput = expectedOutputs[i];

        std::vector<double> output = predict(input);
        for (size_t j = 0; j < output.size(); j++) {
            double error = expectedOutput[j] - output[j];
            sumSquaredError += error * error;
        }
    }
    return sumSquaredError / inputs.size();
}

double Network::accuracy(const std::vector<std::vector<double>>& inputs,
                         const std::vector<std::vector<double>>& expectedOutputs) {
    if (inputs.size() != expectedOutputs.size()) {
        throw std::runtime_error("Mismatch in the number of input and output samples");
    }

    int correctPredictions = 0;
    for (size_t i = 0; i < inputs.size(); i++) {
        const std::vector<double>& input = inputs[i];
        const std::vector<double>& expectedOutput = expectedOutputs[i];

        // predict method returns the output probabilities
        std::vector<double> outputProbabilities = predict(input);
        int predictedClass =
            std::max_element(outputProbabilities.begin(), outputProbabilities.end()) -
            outputProbabilities.begin();

        // expectedOutput is also probabilities, find index of the max
        int expectedClass =
            std::max_element(expectedOutput.begin(), expectedOutput.end()) - expectedOutput.begin();

        if (predictedClass == expectedClass) {
            correctPredictions++;
        }
    }

    return static_cast<double>(correctPredictions) / inputs.size();
}

void Network::train(const std::vector<std::vector<double>>& inputs,
                    const std::vector<std::vector<double>>& outputs, unsigned numEpochs,
                    double learningRate, double regularizationTerm, double validationFraction,
                    TrainingCallback callback) {
    if (inputs.size() != outputs.size()) {
        throw std::runtime_error("Mismatch in the number of input and output samples");
    }

    unsigned numTrainSamples = static_cast<unsigned>(inputs.size() * (1.0 - validationFraction));
    std::vector<std::vector<double>> trainInputs(inputs.begin(), inputs.begin() + numTrainSamples);
    std::vector<std::vector<double>> trainOutputs(outputs.begin(),
                                                  outputs.begin() + numTrainSamples);
    std::vector<std::vector<double>> validationInputs(inputs.begin() + numTrainSamples,
                                                      inputs.end());
    std::vector<std::vector<double>> validationOutputs(outputs.begin() + numTrainSamples,
                                                       outputs.end());

    if (!trainInputs.empty() && trainInputs[0].size() != layers[0].getInputSize()) {
        throw std::runtime_error("Mismatch in the input size and the expected number of inputs");
    }

    std::vector<std::vector<double>> layerDeltas(layers.size());
    auto index = 0;
    for (Layer& layer : layers) {
        layerDeltas[index++].resize(layer.getOutputSize());
    }

    std::random_device rd;
    std::mt19937 g(rd());

    for (unsigned epoch = 0; epoch < numEpochs; epoch++) {
        double totalError = 0.0;

        // Shuffle training samples
        std::vector<size_t> sampleIndices(trainInputs.size());
        for (size_t i = 0; i < sampleIndices.size(); i++) {
            sampleIndices[i] = i;
        }
        std::shuffle(sampleIndices.begin(), sampleIndices.end(), g);

        // Train on each sample
        for (size_t i = 0; i < trainInputs.size(); i++) {
            // Forward pass
            std::vector<double> layerOutputs = trainInputs[sampleIndices[i]];

            // std::cout << "input: ";
            // for (auto i = 0; i < layerOutputs.size(); i++) {
            //     std::cout << layerOutputs[i] << " ";
            // }
            // std::cout << std::endl;
            int k = 0;
            for (Layer& layer : layers) {
                layerOutputs = layer.feedForward(layerOutputs);
            }

            // Calculate error
            double error =
                layers.back().getLossFunction()(trainOutputs[sampleIndices[i]], layerOutputs);
            totalError += error;

            // Backward pass
            for (int layerIndex = layers.size() - 1; layerIndex >= 0; layerIndex--) {
                Layer& layer = layers[layerIndex];

                std::vector<double> output = layer.getOutputs();
                if (layer.withSoftmax()) {
                    // Softmax with cross-entropy loss
                    for (size_t j = 0; j < layer.neurons.size(); j++) {
                        layerDeltas[layerIndex][j] = trainOutputs[sampleIndices[i]][j] - output[j];
                        if (std::isnan(layerDeltas[layerIndex][j])) {
                            throw std::runtime_error("NaN value in layer deltas (softmax)");
                        }
                    }
                } else {
                    if (layerIndex == layers.size() - 1) {
                        // Special case for output layer
                        for (size_t j = 0; j < layer.neurons.size(); j++) {
                            layerDeltas[layerIndex][j] =
                                (trainOutputs[sampleIndices[i]][j] - output[j]) *
                                layer.activationDerivativeFunction(output[j]);
                            if (std::isnan(layerDeltas[layerIndex][j])) {
                                throw std::runtime_error("NaN value in layer deltas (output)");
                            }
                        }
                    } else {  // Hidden layers
                        auto layerDeltaNextLayer = layerDeltas[layerIndex + 1];

                        for (size_t k = 0; k < layer.neurons.size(); k++) {
                            double sum = 0.0;

                            for (size_t j = 0; j < layer.getInputSize(); j++) {
                                sum += layer.neurons[k].getWeight(j) * layerDeltaNextLayer[j];

                                if (std::isnan(sum)) {
                                    std::cout << "Layer index: " << layerIndex << std::endl;
                                    std::cout << "Neuron index: " << k << std::endl;
                                    std::cout << "Weight: " << layer.neurons[k].getWeight(j)
                                              << std::endl;
                                    std::cout
                                        << "Layer delta next layer: " << layerDeltaNextLayer[j]
                                        << std::endl;
                                    throw std::runtime_error("NaN value in sum (hidden)");
                                }
                            }

                            double activationDerivative =
                                layer.activationDerivativeFunction(output[k]);
                            // std::cout << "Activation derivative: " << activationDerivative
                            //           << " | Output: " << output[k]
                            //           << " | Layer index: " << layerIndex << std::endl;
                            layerDeltas[layerIndex][k] = sum * activationDerivative;

                            // std::cout << "Layer index: " << layerIndex << std::endl;
                            // std::cout << "Neuron index: " << k << std::endl;
                            // std::cout << "Weights: ";
                            // for (size_t j = 0; j < layer.getInputSize(); j++) {
                            //     std::cout << layer.neurons[k].getWeight(j) << " ";
                            // }
                            // std::cout << std::endl;
                            // std::cout << "Next layer deltas: ";
                            // for (size_t j = 0; j < layerDeltaNextLayer.size(); j++) {
                            //     std::cout << layerDeltaNextLayer[j] << " ";
                            // }
                            // std::cout << std::endl;
                            // std::cout << "Layer delta: " << layerDeltas[layerIndex][k] <<
                            // std::endl; std::cout << "Sum: " << sum << std::endl; std::cout <<
                            // "Activation derivative: " << activationDerivative
                            //           << std::endl;
                            if (std::isnan(layerDeltas[layerIndex][k])) {
                                throw std::runtime_error("NaN value in layer deltas (hidden)");
                            }
                        }
                    }
                }

                // std::cout << layerIndex << ") Layer delta: ";
                // for (auto i = 0; i < layerDeltas[layerIndex].size(); i++) {
                //     std::cout << layerDeltas[layerIndex][i] << " ";
                // }
                // std::cout << std::endl;

                // Update weights
                layer.adjustWeights(layer.getInputValues(), layerDeltas[layerIndex], learningRate,
                                    regularizationTerm);
            }
        }
        totalError /= trainInputs.size();

        double validationError = 0.0;
        if (!validationInputs.empty()) {
            validationError = validate(validationInputs, validationOutputs);
        }

        if (callback) {
            callback(epoch, totalError, validationError);
        }
    }
}

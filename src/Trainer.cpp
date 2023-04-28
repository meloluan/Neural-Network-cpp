#include "Trainer.h"

#include <fstream>

Trainer::Trainer(Network network, std::vector<std::vector<double>> trainingInputs,
                 std::vector<std::vector<double>> trainingOutputs)
        : m_network(network),
          m_trainingInputs(trainingInputs),
          m_trainingOutputs(trainingOutputs) {}

// train the network using backpropagation
void Trainer::train(double learningRate, int numEpochs) {
    m_network.train(m_trainingInputs, m_trainingOutputs, learningRate, numEpochs);
}

// calculate the network's output given input values
std::vector<double> Trainer::predict(std::vector<double> inputs) {
    return m_network.predict(inputs);
}

// save the network to a file
// void Trainer::save(std::string filename) {
//     std::ofstream file(filename, std::ios::binary);
//     if (file.is_open()) {
//         // write the number of layers to the file
//         int num_layers = network.getNumLayers();
//         file.write(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));

//         // write each layer to the file
//         for (int i = 0; i < num_layers; i++) {
//             Layer layer = network.getLayer(i);

//             // write the layer's activation function to the file
//             ActivationFunction activation_function = layer.getActivationFunction();
//             file.write(reinterpret_cast<char*>(&activation_function),
//             sizeof(activation_function));

//             // write the layer's weights to the file
//             std::vector<double> weights = layer.getWeights();
//             int num_weights = weights.size();
//             file.write(reinterpret_cast<char*>(&num_weights), sizeof(num_weights));
//             file.write(reinterpret_cast<char*>(&weights[0]), num_weights * sizeof(double));
//         }

//         file.close();
//     }
// }

// // load the network from a file
// void Trainer::load(std::string filename) {
//     std::ifstream file(filename, std::ios::binary);
//     if (file.is_open()) {
//         // read the number of layers from the file
//         int num_layers;
//         file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));

//         // read each layer from the file
//         std::vector<int> num_neurons(num_layers);
//         std::vector<ActivationFunction> activation_functions(num_layers);
//         std::vector<std::vector<double>> weights(num_layers);
//         for (int i = 0; i < num_layers; i++) {
//             // read the layer's activation function from the file
//             ActivationFunction activation_function;
//             file.read(reinterpret_cast<char*>(&activation_function),
//             sizeof(activation_function)); activation_functions[i] = activation_function;

//             // read the layer's weights from the file
//             int num_weights;
//             file.read(reinterpret_cast<char*>(&num_weights), sizeof(num_weights));
//             std::vector<double> layer_weights(num_weights);
//             file.read(reinterpret_cast<char*>(&layer_weights[0]), num_weights * sizeof(double));
//             weights[i] = layer_weights;
//             num_neurons[i] = layer_weights.size() / (i == 0 ? 1 : weights[i - 1].size());
//         }

//         // initialize the network with the loaded layers
//         network.initialize(num_neurons[0], num_neurons.back(), num_neurons);
//         for (int i = 0; i < num_layers; i++) {
//             Layer layer(activation_functions[i]);
//             layer.setWeights(weights[i]);
//             network.setLayer(i, layer);
//         }

//         file.close();
//     }
// }
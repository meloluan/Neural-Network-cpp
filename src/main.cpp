#include <iostream>
#include <vector>

#include "Network.h"
#include "Trainer.h"
#include "Utils.h"
#include "mnist/mnist_reader.hpp"
#include "preprocess/Pca.h"

using namespace std;

// Shuffle the dataset
void shuffle(int *array, size_t n) {
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++) {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}
int main() {
    // MNIST_DATA_LOCATION set by MNIST cmake config
    // std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

    // // Load MNIST data
    // mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
    //     mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

    // std::cout << "Number of training images = " << dataset.training_images.size() << std::endl;
    // std::cout << "Number of training labels = " << dataset.training_labels.size() << std::endl;
    // std::cout << "Number of test images = " << dataset.test_images.size() << std::endl;
    // std::cout << "Number of test labels = " << dataset.test_labels.size() << std::endl;

    // // Convertendo as imagens para um vetor de double entre 0 e 1
    // vector<vector<double>> inputs;
    // for (auto& image : dataset.training_images) {
    //     inputs.push_back(Utils::normalize(Utils::toVector(image)));
    // }

    // int numOutputs = 10;

#define numInputs 2
#define numHiddenNodes 2
#define numOutputs 1
#define numTrainingSets 4

    std::vector<std::vector<double>> inputs = {
        {0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}};
    std::vector<std::vector<double>> outputs = {{0.0f}, {1.0f}, {1.0f}, {0.0f}};

    // Convertendo as saídas esperadas para um vetor de double
    // vector<vector<double>> outputs(dataset.training_labels.size(), vector<double>(numOutputs,
    // 0.0));

    // Convertendo as saídas esperadas para um vetor de 10 posições, onde cada posição é 0 ou 1
    // for (int i = 0; i < dataset.training_labels.size(); i++) {
    //     outputs[i][dataset.training_labels[i]] = 1.0;
    // }

    // Pca preprocessor(50);
    // inputs = preprocessor.process(inputs);

    // std::cout << "Number of dimensions = " << inputs[0].size() << std::endl;

    // cria uma rede neural com 50 entradas, 1 camadas oculta com 64 neurônios, e 10 saídas
    Network network;
    network.initialize(2, numOutputs, {2, numOutputs});

    Trainer trainer(network, inputs, outputs);

    // train the network for 1000 epochs with a learning rate of 0.1
    trainer.train(0.1, 1000);

    // Testando a rede neural com as imagens de teste
    int numCorrect = 0;
    // for (size_t i = 0; i < dataset.test_images.size(); i++) {
    //     vector<double> input = Utils::normalize(Utils::toVector(dataset.test_images[i]));
    //     vector<double> output = network.predict(input);
    //     int predictedLabel = Utils::maxIndex(output);
    //     int trueLabel = dataset.test_labels[i];
    //     if (predictedLabel == trueLabel) {
    //         numCorrect++;
    //     }
    // }

    // cout << "Acurácia: " << numCorrect / (double)dataset.test_images.size() << endl;

    return 0;
}

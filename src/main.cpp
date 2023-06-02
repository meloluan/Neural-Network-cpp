#include <chrono>
#include <iostream>
#include <vector>

#include "Network.h"
#include "Utils.h"
#include "matplotlibcpp.h"
#include "mnist/mnist_reader.hpp"
#include "preprocess/Pca.h"

namespace plt = matplotlibcpp;
using namespace std;

void mnist_digits_problem();
void xor_problem();
void performParameterSearch(
    const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& outputs,
    const std::vector<std::pair<unsigned, ActivationFunctionType>>& layerConfig,
    LossFunctionType lossFunction, const std::vector<unsigned>& numEpochsValues,
    const std::vector<double>& learningRateValues,
    const std::vector<double>& validationFractionValues, unsigned numFolds);

template <typename T>
double crossValidate(Network& nn, unsigned numEpochs, double learningRate,
                     double regularizationTerm, double validationFraction,
                     const std::vector<std::vector<T>>& inputs,
                     const std::vector<std::vector<T>>& outputs, unsigned k);

int main() {
    // mnist_digits_problem();
    xor_problem();
    return 0;
}

void mnist_digits_problem() {
    // MNIST_DATA_LOCATION set by MNIST cmake config
    std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

    // Load MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

    std::cout << "Number of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "Number of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "Number of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "Number of test labels = " << dataset.test_labels.size() << std::endl;

    // Convertendo as imagens para um vetor de double entre 0 e 1
    std::vector<std::vector<double>> inputs;
    for (auto& image : dataset.training_images) {
        std::vector<double> normalizedImage;
        for (auto pixel : image) {
            normalizedImage.push_back(static_cast<double>(pixel) / 255.0);
        }
        inputs.push_back(normalizedImage);
    }

    int numOutputs = 10;

    // Convertendo as saídas esperadas para um vetor de double
    vector<vector<double>> outputs(dataset.training_labels.size(), vector<double>(numOutputs, 0.0));

    // Convertendo as saídas esperadas para um vetor de 10 posições, onde cada posição é 0 ou 1
    for (int i = 0; i < dataset.training_labels.size(); i++) {
        outputs[i][dataset.training_labels[i]] = 1.0;
    }

    // Pca preprocessor(100);
    // inputs = preprocessor.process(inputs);

    std::cout << "Number of dimensions = " << inputs[0].size() << std::endl;

    // Criar a rede neural
    std::vector<std::pair<unsigned, ActivationFunctionType>> layerConfig = {
        {784, ActivationFunctionType::Sigmoid},
        {256, ActivationFunctionType::Sigmoid},
        {10, ActivationFunctionType::Softmax}};

    // std::vector<std::pair<unsigned, ActivationFunctionType>> layerConfig = {
    //     {100, ActivationFunctionType::ReLU},
    //     {256, ActivationFunctionType::ReLU},
    //     {10, ActivationFunctionType::Softmax}};

    // std::vector<double> learningRates = {0.1, 0.01, 0.001, 0.0001, 0.00001};
    // std::vector<double> validationFractions = {0.1, 0.2, 0.3, 0.4, 0.5};
    // std::vector<unsigned> numEpochs = {5, 10, 50};

    std::vector<unsigned> numEpochs = {10};
    std::vector<double> learningRates = {0.001};
    std::vector<double> validationFractions = {0.1};

    // // Create a new neural network and train it on the training set
    // Network nn(layerConfig, LossFunctionType::CrossEntropy);
    // nn.train(inputs, outputs, numEpochs, learningRate, validationFraction,
    //          [&numEpochs, &validationFraction](unsigned epoch, double error, double valError) {
    //              epoch++;
    //              static auto startTime = std::chrono::high_resolution_clock::now();
    //              auto currentTime = std::chrono::high_resolution_clock::now();
    //              double elapsedTime =
    //                  std::chrono::duration<double>(currentTime - startTime).count();

    //              double remainingTime = (numEpochs - epoch) * elapsedTime / epoch;

    //              std::cout << "Epoch " << epoch << " - Error: " << error
    //                        << " - Validation Error: " << validationFraction
    //                        << " - Elapsed Time: " << elapsedTime
    //                        << "s - Remaining Time: " << remainingTime << "s" << std::endl;
    //          });

    // double correct = 0;
    // for (size_t i = 0; i < inputs.size(); i++) {
    //     auto output = nn.predict(inputs[i]);
    //     auto prediction = round(output[0]);
    //     if (prediction == outputs[i][0]) {
    //         correct++;
    //     }
    // }
    // std::cout << "Correct: " << correct << std::endl;
    // double accuracy = correct / inputs.size() * 100;

    // std::cout << "Accuracy: " << accuracy << "%" << std::endl;
    performParameterSearch(inputs, outputs, layerConfig, LossFunctionType::CrossEntropy, numEpochs,
                           learningRates, validationFractions, 2);
}
void performParameterSearch(
    const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& outputs,
    const std::vector<std::pair<unsigned, ActivationFunctionType>>& layerConfig,
    LossFunctionType lossFunction, const std::vector<unsigned>& numEpochsValues,
    const std::vector<double>& learningRateValues,
    const std::vector<double>& validationFractionValues, unsigned numFolds) {
    std::vector<double> bestParams = {0.0, 0.0, 0.0};
    double bestAccuracy = 0.0;

    // Loop through all combinations of parameter values
    for (double numEpochs : numEpochsValues) {
        for (double learningRate : learningRateValues) {
            for (double validationFraction : validationFractionValues) {
                // Calculate accuracy for this parameter combination using cross-validation
                double accuracy = 0.0;

                // Perform cross-validation
                Network nn(layerConfig, lossFunction);

                auto regularizationTerm = 0.1;
                accuracy = crossValidate(nn, numEpochs, learningRate, regularizationTerm,
                                         validationFraction, inputs, outputs, numFolds);
                std::cout << "Accuracy: " << accuracy << std::endl;

                // Check if this is the best parameter combination so far
                if (accuracy >= bestAccuracy) {
                    bestAccuracy = accuracy;
                    bestParams = {numEpochs, learningRate, validationFraction};
                }
            }
        }
    }

    // Print the best parameter combination found
    std::cout << "Best parameters: numEpochs=" << bestParams[0]
              << ", learningRate=" << bestParams[1] << ", validationFraction=" << bestParams[2]
              << ", accuracy=" << bestAccuracy << std::endl;
}

template <typename T>
double crossValidate(Network& nn, unsigned numEpochs, double learningRate,
                     double regularizationTerm, double validationFraction,
                     const std::vector<std::vector<T>>& inputs,
                     const std::vector<std::vector<T>>& outputs, unsigned k) {
    // Define o tamanho das partes do dataset
    size_t partSize = inputs.size() / k;
    // Cria vetores para armazenar as métricas de avaliação em cada parte
    std::vector<double> metrics(k);
    // Itera k vezes
    for (unsigned i = 0; i < k; i++) {
        // Divide o dataset em partes de treinamento e teste
        size_t startIdx = i * partSize;
        size_t endIdx = (i == k - 1) ? inputs.size() : (i + 1) * partSize;
        std::cout << "startIdx: " << startIdx << " - endIdx: " << endIdx << std::endl;
        std::vector<std::vector<T>> trainInputs, trainOutputs, testInputs, testOutputs;
        for (size_t j = 0; j < inputs.size(); j++) {
            if (j >= startIdx && j < endIdx) {
                testInputs.push_back(inputs[j]);
                testOutputs.push_back(outputs[j]);
            } else {
                trainInputs.push_back(inputs[j]);
                trainOutputs.push_back(outputs[j]);
            }
        }

        // Treina o modelo com as partes de treinamento
        nn.train(trainInputs, trainOutputs, numEpochs, learningRate, regularizationTerm,
                 validationFraction,
                 [&numEpochs, &validationFraction](unsigned epoch, double error, double valError) {
                     epoch++;
                     static auto startTime = std::chrono::high_resolution_clock::now();
                     auto currentTime = std::chrono::high_resolution_clock::now();
                     double elapsedTime =
                         std::chrono::duration<double>(currentTime - startTime).count();

                     double remainingTime = (numEpochs - epoch) * elapsedTime / epoch;

                     std::cout << "Epoch " << epoch << " - Error: " << error
                               << " - Validation Error: " << validationFraction
                               << " - Elapsed Time: " << elapsedTime
                               << "s - Remaining Time: " << remainingTime << "s" << std::endl;
                 });
        // Testa o modelo na parte de teste e armazena a métrica de avaliação
        double metric = nn.accuracy(testInputs, testOutputs);
        std::cout << "Metric: " << metric << std::endl;
        metrics[i] = metric;
    }
    // Calcula a média das métricas de avaliação em cada parte e retorna
    double meanMetric = std::accumulate(metrics.begin(), metrics.end(), 0.0) / k;
    return meanMetric;
}

void xor_problem() {
    std::vector<std::vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> outputs = {{0}, {1}, {1}, {0}};

    std::vector<std::pair<unsigned, ActivationFunctionType>> layerConfig = {
        {2, ActivationFunctionType::Sigmoid},
        {2, ActivationFunctionType::Sigmoid},
        {1, ActivationFunctionType::Sigmoid}};

    auto numEpochs = 10000;
    auto learningRate = 0.01;
    auto validationFraction = 0.0;
    auto regularizationTerm = 0.0;

    std::vector<double> trainLossHistory;
    std::vector<double> valLossHistory;

    // Create a new neural network and train it on the training set
    Network nn(layerConfig, LossFunctionType::BinaryCrossEntropy);
    nn.train(
        inputs, outputs, numEpochs, learningRate, regularizationTerm, validationFraction,
        [&numEpochs, &validationFraction, &trainLossHistory, &valLossHistory](
            unsigned epoch, double error, double valError) {
            epoch++;
            static auto startTime = std::chrono::high_resolution_clock::now();
            auto currentTime = std::chrono::high_resolution_clock::now();
            double elapsedTime = std::chrono::duration<double>(currentTime - startTime).count();

            double remainingTime = (numEpochs - epoch) * elapsedTime / epoch;

            trainLossHistory.push_back(error);
            valLossHistory.push_back(valError);

            std::cout << "Epoch " << epoch << " - Error: " << error
                      << " - Validation Error: " << valError << " - Elapsed Time: " << elapsedTime
                      << "s - Remaining Time: " << remainingTime << "s" << std::endl;
        });

    plt::figure();
    plt::title("Training Loss");
    plt::xlabel("Epoch");
    plt::ylabel("Loss");
    plt::plot(trainLossHistory);
    plt::legend();  // mostra a legenda
    std::cout << "Saving training loss plot on ./trainLoss.png" << std::endl;
    plt::savefig("./trainLoss.png");

    plt::figure();
    plt::title("Validation Loss");
    plt::xlabel("Epoch");
    plt::ylabel("Loss");
    plt::plot(valLossHistory);
    plt::legend();  // mostra a legenda
    std::cout << "Saving validation loss plot on ./valLoss.png" << std::endl;
    plt::savefig("./valLoss.png");

    double correct = 0;
    for (size_t i = 0; i < inputs.size(); i++) {
        auto output = nn.predict(inputs[i]);
        auto prediction = round(output[0]);
        if (prediction == outputs[i][0]) {
            correct++;
        }
    }
    double accuracy = correct / inputs.size() * 100;

    std::cout << "Accuracy: " << accuracy << "%" << std::endl;

    // Testando a rede
    for (const auto& input : inputs) {
        auto output = nn.predict(input);
        std::cout << "Input: " << input[0] << ", " << input[1] << "\n";
        std::cout << "Output: " << output[0] << "\n";
    }
}
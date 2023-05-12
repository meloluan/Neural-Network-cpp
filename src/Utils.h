#include <math.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace Utils {

static double getRandom() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<double> dis(0.0, 1.0);
    return dis(gen);
}

// Normaliza um vetor de dados para ter média 0 e desvio padrão 1
static std::vector<double> normalize(const std::vector<double>& data) {
    double mean = 0.0, sd = 0.0;

    // Calcula a média
    for (auto& d : data) {
        mean += d;
    }
    mean /= data.size();

    // Calcula o desvio padrão
    for (auto& d : data) {
        sd += std::pow(d - mean, 2);
    }
    sd = std::sqrt(sd / (data.size() - 1));

    // Normaliza os dados
    std::vector<double> normalizedData(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        normalizedData[i] = (data[i] - mean) / sd;
    }

    return normalizedData;
}

// Converte uma imagem 28x28 em um vetor de tamanho 784
static std::vector<double> toVector(const std::vector<uint8_t>& image) {
    std::vector<double> vec(784);
    for (size_t i = 0; i < 784; i++) {
        vec[i] = static_cast<double>(image[i]) / 255.0;
    }
    return vec;
}

// Retorna o índice do maior valor de um vetor
static size_t maxIndex(const std::vector<double>& output) {
    double maxVal = output[0];
    size_t maxIndex = 0;
    for (size_t i = 1; i < output.size(); i++) {
        if (output[i] > maxVal) {
            maxVal = output[i];
            maxIndex = i;
        }
    }
    return maxIndex;
}

static double mse(std::vector<double> expected, std::vector<double> actual) {
    if (expected.size() != actual.size()) {
        throw std::invalid_argument("Expected and actual vectors must have same size.");
    }
    double error_sum = 0.0;
    for (int i = 0; i < expected.size(); i++) {
        double error = expected[i] - actual[i];
        error_sum += error * error;
    }
    double mse = error_sum / expected.size();
    return mse;
}

static void splitData(const std::vector<std::vector<double>>& inputs,
                      const std::vector<std::vector<double>>& outputs, double validationFraction,
                      std::vector<std::vector<double>>& trainInputs,
                      std::vector<std::vector<double>>& trainOutputs,
                      std::vector<std::vector<double>>& valInputs,
                      std::vector<std::vector<double>>& valOutputs) {
    // Determinar o número de exemplos para validação
    size_t numValExamples = static_cast<size_t>(inputs.size() * validationFraction);

    // Selecionar aleatoriamente os exemplos de validação
    std::vector<size_t> valIndices;
    std::unordered_set<size_t> valIndicesSet;
    while (valIndicesSet.size() < numValExamples) {
        size_t index = rand() % inputs.size();
        if (valIndicesSet.find(index) == valIndicesSet.end()) {
            valIndices.push_back(index);
            valIndicesSet.insert(index);
        }
    }

    // Separar os exemplos de treinamento e validação
    trainInputs.reserve(inputs.size() - numValExamples);
    trainOutputs.reserve(outputs.size() - numValExamples);
    valInputs.reserve(numValExamples);
    valOutputs.reserve(numValExamples);
    for (size_t i = 0; i < inputs.size(); i++) {
        if (valIndicesSet.find(i) != valIndicesSet.end()) {
            valInputs.push_back(inputs[i]);
            valOutputs.push_back(outputs[i]);
        } else {
            trainInputs.push_back(inputs[i]);
            trainOutputs.push_back(outputs[i]);
        }
    }
}
static double testAccuracy(Network& nn, const std::vector<std::vector<double>>& inputs,
                           const std::vector<std::vector<double>>& expectedOutputs) {
    unsigned numCorrect = 0;
    for (size_t i = 0; i < inputs.size(); i++) {
        std::vector<double> output = nn.predict(inputs[i]);
        int predictedLabel = Utils::maxIndex(output);
        int trueLabel = Utils::maxIndex(expectedOutputs[i]);
        if (predictedLabel == trueLabel) {
            numCorrect++;
        }
    }
    return (double)numCorrect / inputs.size();
}

};  // namespace Utils
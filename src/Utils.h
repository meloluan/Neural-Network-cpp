#include <math.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace Utils {

static double getRandom() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<double> dis(0.0, 1.0);
    return dis(gen);
}
static double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}
static double sigmoidDerivative(double x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

static std::vector<double> softmax(const std::vector<double>& x) {
    double max = *std::max_element(x.begin(), x.end());
    std::vector<double> exps(x.size());
    double expsSum = 0;
    for (int i = 0; i < x.size(); i++) {
        exps[i] = std::exp(x[i] - max);
        expsSum += exps[i];
    }
    std::vector<double> result(x.size());
    for (int i = 0; i < x.size(); i++) {
        result[i] = exps[i] / expsSum;
    }
    return result;
}

static double softmaxDerivative(std::vector<double> outputs, int index) {
    double softmax = Utils::softmax(outputs)[index];
    double delta;
    for (int i = 0; i < outputs.size(); i++) {
        if (i == index) {
            delta += softmax * (1 - softmax);
        } else {
            delta -= softmax * Utils::softmax(outputs)[i];
        }
    }
    return delta;
}
static double relu(double x) {
    return std::max(0.0, x);
}

static double reluDerivative(double x) {
    return x > 0 ? 1 : 0;
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

};  // namespace Utils
#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace Functions {
// Funções de ativação e suas derivadas
static double sigmoid(double x) {
    auto ret = 1.0 / (1.0 + std::exp(-x));

    if (std::isnan(ret)) {
        std::cout << "x = " << x << "\n";
        // throw std::runtime_error("NaN value in sigmoid");
    }
    return ret;
}

static double sigmoidDerivative(double x) {
    double sig = sigmoid(x);
    auto ret = sig * (1 - sig);

    if (std::isnan(ret)) {
        throw std::runtime_error("NaN value in sigmoid derivative");
    }

    return ret;
}

static double reLU(double x) {
    return std::max(0.0, x);
}

static double reLUDerivative(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

static std::vector<double> softmax(const std::vector<double>& x) {
    double maxElem = *std::max_element(x.begin(), x.end());
    std::vector<double> exps(x.size());
    double expSum = 0.0;

    for (size_t i = 0; i < x.size(); i++) {
        exps[i] = std::exp(x[i] - maxElem);
        expSum += exps[i];
    }

    for (double& e : exps) {
        e /= expSum;

        if (std::isnan(e)) {
            throw std::runtime_error("NaN value in softmax");
        }
    }

    return exps;
}

static std::vector<double> softmaxDerivative(const std::vector<double>& x) {
    std::vector<double> softm = softmax(x);
    std::vector<double> derivative(x.size());

    for (size_t i = 0; i < x.size(); i++) {
        derivative[i] = softm[i] * (1 - softm[i]);

        if (std::isnan(derivative[i])) {
            throw std::runtime_error("NaN value in softmax derivative");
        }
    }

    return derivative;
}
}  // namespace Functions

#endif  // ACTIVATION_FUNCTIONS_H

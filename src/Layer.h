#ifndef LAYER_H
#define LAYER_H

#include <functional>
#include <vector>

#include "ActivationFunctions.h"
#include "Neuron.h"

enum class ActivationFunctionType { Sigmoid, ReLU, Softmax };

enum class LossFunctionType { None, MeanSquaredError, CrossEntropy, BinaryCrossEntropy };

class Layer {
public:
    using Activation = std::function<double(double)>;
    using LossFunction =
        std::function<double(const std::vector<double>&, const std::vector<double>&)>;

    /**
     * @brief Constructs a new Layer object.
     *
     * @param inputSize The size of the input to the layer.
     * @param outputSize The size of the output from the layer.
     * @param activationFunction The activation function to be used by the layer.
     * @param lossFunction The loss function to be used for the layer (optional).
     */
    Layer(unsigned int inputSize, unsigned int outputSize,
          ActivationFunctionType activationFunction,
          LossFunctionType lossFunction = LossFunctionType::None);

    /**
     * @brief Performs feed-forward computation for the layer.
     *
     * @param inputs The input values for the layer.
     * @return std::vector<double> The output values of the layer.
     */
    std::vector<double> feedForward(const std::vector<double>& inputs);

    /**
     * @brief Adjusts the weights of the neurons in the layer based on the error deltas.
     *
     * @param inputs The input values for the layer.
     * @param deltas The error deltas for the layer.
     * @param learningRate The learning rate for weight adjustment.
     * @param regularizationTerm The regularization term used for weight adjustment.
     */
    void adjustWeights(const std::vector<double>& inputs, const std::vector<double>& deltas,
                       double learningRate, double regularizationTerm);

    /**
     * @brief Computes the output of the activation function for a given input.
     *
     * @param input The input value to the activation function.
     * @return double The output value of the activation function.
     */
    double activationFunction(double input);

    /**
     * @brief Computes the derivative of the activation function for a given input.
     *
     * @param input The input value to the activation function.
     * @return double The derivative value of the activation function.
     */
    double activationDerivativeFunction(double input);

    std::vector<Neuron> neurons;  ///< The neurons in the layer.

    /**
     * @brief Returns the output values of the layer.
     *
     * @return std::vector<double> The output values of the layer.
     */
    std::vector<double> getOutputs();

    /**
     * @brief Returns the input values of the layer.
     *
     * @return std::vector<double> The input values of the layer.
     */
    std::vector<double> getInputValues() const;

    /**
     * @brief Returns the loss function used by the layer.
     *
     * @return LossFunction The loss function used by the layer.
     */
    LossFunction getLossFunction();

    /**
     * @brief Returns the size of the output from the layer.
     *
     * @return size_t The size of the output from the layer.
     */
    size_t getOutputSize();

    /**
     * @brief Returns the size of the input to the layer.
     *
     * @return size_t The size of the input to the layer.
     */
    size_t getInputSize();

    /**
     * @brief Returns the weights of the neurons in the layer.
     *
     * @return std::vector<std::vector<double>> The weights of the neurons in the layer.
     */
    std::vector<std::vector<double>> getWeights();

    /**
     * @brief Checks if the layer uses the softmax activation function.
     *
     * @return bool True if the layer uses softmax, false otherwise.
     */
    bool withSoftmax();

private:
    Activation activation;            ///< The activation function of the layer.
    Activation activationDerivative;  ///< The derivative of the activation function of the layer.

    LossFunctionType m_lossFunction =
        LossFunctionType::None;  ///< The loss function used by the layer.

    bool m_withSoftmax = false;  ///< Flag indicating if the layer uses softmax activation.

    /**
     * @brief Retrieves the activation and derivative functions for a given activation function
     * type.
     *
     * @param activationFunction The activation function type.
     * @return std::pair<Activation, Activation> The activation and derivative functions.
     */
    std::pair<Activation, Activation> getActivationFunctions(
        ActivationFunctionType activationFunction);

    std::vector<double> m_outputs;      ///< The output values of the layer.
    std::vector<double> m_inputValues;  ///< The input values of the layer.
};

#endif  // LAYER_H

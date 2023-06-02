#include <functional>
#include <initializer_list>
#include <vector>

#include "Layer.h"

enum class OptimizationMethod { GradientDescent, Momentum, Adam };

class Network {
public:
    using TrainingCallback =
        std::function<void(unsigned epoch, double error, double validationError)>;

    /**
     * @brief Constructs a new Network object.
     *
     * @param layerConfig The configuration of the network layers, specifying the number of neurons
     * in each layer and the activation function type.
     * @param lossFunction The loss function used by the network (optional).
     */
    Network(std::vector<std::pair<unsigned int, ActivationFunctionType>> layerConfig,
            LossFunctionType lossFunction = LossFunctionType::None);

    /**
     * @brief Sets the weights of the hidden and output layers.
     *
     * @param hiddenLayerWeights The weights of the hidden layer neurons.
     * @param outputLayerWeights The weights of the output layer neurons.
     */
    void setWeights(const std::vector<std::vector<double>>& hiddenLayerWeights,
                    const std::vector<std::vector<double>>& outputLayerWeights);

    /**
     * @brief Performs a forward pass through the network to predict the output for a given input.
     *
     * @param inputs The input values for the network.
     * @return std::vector<double> The predicted output values.
     */
    std::vector<double> predict(const std::vector<double>& inputs);

    /**
     * @brief Calculates the validation error for the network.
     *
     * @param inputs The input samples for validation.
     * @param outputs The expected output samples for validation.
     * @return double The validation error.
     */
    double validate(const std::vector<std::vector<double>>& inputs,
                    const std::vector<std::vector<double>>& outputs);

    /**
     * @brief Calculates the mean squared error for the network.
     *
     * @param inputs The input samples.
     * @param expectedOutputs The expected output samples.
     * @return double The mean squared error.
     */
    double meanSquaredError(const std::vector<std::vector<double>>& inputs,
                            const std::vector<std::vector<double>>& expectedOutputs);

    /**
     * @brief Calculates the accuracy of the network predictions.
     *
     * @param inputs The input samples.
     * @param expectedOutputs The expected output samples.
     * @return double The accuracy of the network predictions.
     */
    double accuracy(const std::vector<std::vector<double>>& inputs,
                    const std::vector<std::vector<double>>& expectedOutputs);

    /**
     * @brief Trains the network using the provided training data.
     *
     * @param inputs The input samples for training.
     * @param outputs The expected output samples for training.
     * @param numEpochs The number of training epochs.
     * @param learningRate The learning rate for weight adjustment.
     * @param regularizationTerm The regularization term for weight adjustment.
     * @param validationFraction The fraction of data to be used for validation.
     * @param callback The training callback function (optional).
     */
    void train(const std::vector<std::vector<double>>& inputs,
               const std::vector<std::vector<double>>& outputs, unsigned numEpochs,
               double learningRate, double regularizationTerm, double validationFraction,
               TrainingCallback callback);

private:
    std::vector<Layer> layers;              ///< The layers of the network.
    OptimizationMethod optimizationMethod;  ///< The optimization method used for training.
};

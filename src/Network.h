#include <functional>
#include <initializer_list>
#include <vector>

#include "Layer.h"

enum class OptimizationMethod { GradientDescent, Momentum, Adam };

class Network {
public:
    using TrainingCallback =
        std::function<void(unsigned epoch, double error, double validationError)>;

    Network(std::vector<std::pair<unsigned int, ActivationFunctionType>> layerConfig,
            LossFunctionType lossFunction = LossFunctionType::None);

    void setWeights(const std::vector<std::vector<double>>& hiddenLayerWeights,
                    const std::vector<std::vector<double>>& outputLayerWeights);

    std::vector<double> predict(const std::vector<double>& inputs);

    double validate(const std::vector<std::vector<double>>& inputs,
                    const std::vector<std::vector<double>>& outputs);

    double meanSquaredError(const std::vector<std::vector<double>>& inputs,
                            const std::vector<std::vector<double>>& expectedOutputs);

    void train(const std::vector<std::vector<double>>& inputs,
               const std::vector<std::vector<double>>& outputs, unsigned numEpochs,
               double learningRate, double validationFraction, TrainingCallback callback);

private:
    std::vector<Layer> layers;
    OptimizationMethod optimizationMethod;

    void updateWeights(double learningRate, const std::vector<double>& hiddenGradients,
                       const std::vector<double>& outputGradients);
};

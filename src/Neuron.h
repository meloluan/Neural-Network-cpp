
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

/**
 * This class represents a single Neuron in a Neural Network Layer.
 * It contains weights, bias and the functionalities to adjust them.
 */
class Neuron {
public:
    /**
     * Construct a new Neuron object
     *
     * @param numInputs The number of input connections to the neuron
     */
    Neuron(unsigned numInputs);

    /**
     * Feed-forward operation for a neuron. It calculates the weighted sum of inputs and bias.
     *
     * @param inputs Input values
     * @return double Weighted sum of inputs and bias
     */
    double feedForward(const std::vector<double>& inputs) const;

    /**
     * Adjusts the weights and bias of the neuron based on the error, inputs and learning rate.
     *
     * @param inputs Input values
     * @param delta The error delta
     * @param learningRate The learning rate for weight adjustment
     * @param regularizationTerm The regularization term used for L2 regularization
     */
    void adjustWeights(const std::vector<double>& inputs, double delta, double learningRate,
                       double regularizationTerm);

    /**
     * Sets the bias of the neuron to a specific value
     *
     * @param bias The new bias value
     */
    void setBias(double bias);

    /**
     * Gets the weight at a specific index
     *
     * @param index The index of the weight
     * @return double The weight value at the given index
     */
    double getWeight(unsigned index) const;

    /**
     * Gets the weights of the neuron
     *
     * @return std::vector<double> The weights of the neuron
     */
    std::vector<double> getWeights() const;

private:
    std::vector<double> m_weights;  ///< The weights of the neuron
    double m_bias;                  ///< The bias of the neuron
};
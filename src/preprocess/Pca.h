
#include <Eigen/Dense>

#include "Preprocessor.h"

/**
 * This class is a concrete implementation of the Preprocessor interface.
 * It uses Principal Component Analysis (PCA) for dimensionality reduction.
 */
class Pca : public Preprocessor {
public:
    /**
     * Construct a new Pca object
     *
     * @param num_components The number of principal components to keep
     */
    Pca(int num_components);

    /**
     * Applies Principal Component Analysis (PCA) to the input data.
     *
     * @param input The input data
     * @return std::vector<std::vector<double>> The data transformed to the principal component axes
     */
    std::vector<std::vector<double>> process(
        const std::vector<std::vector<double>>& input) override;

private:
    int num_components_;  ///< The number of principal components to keep
    Eigen::VectorXd mean_;
    Eigen::MatrixXd eigenvectors_;
};
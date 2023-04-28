
#include <Eigen/Dense>

#include "Preprocessor.h"

class Pca : public Preprocessor {
public:
    Pca(int num_components);

    std::vector<std::vector<double>> process(
        const std::vector<std::vector<double>>& input) override;

private:
    int num_components_;
    Eigen::VectorXd mean_;
    Eigen::MatrixXd eigenvectors_;
};
#include "Pca.h"

#include <iostream>
#include <vector>

Pca::Pca(int num_components) : num_components_(num_components) {}

std::vector<std::vector<double>> Pca::process(const std::vector<std::vector<double>>& input) {
    int m = input[0].size();  // número de colunas
    int n = input.size();     // número de linhas

    std::cout << "Converting input to Eigen matrix..." << std::endl;
    // Converte a matriz de entrada em uma matriz do Eigen
    Eigen::MatrixXd X(n, m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            X(i, j) = input[i][j];
        }
    }

    std::cout << "Calculating transformation matrix..." << std::endl;

    // Centraliza a matriz de entrada
    Eigen::VectorXd mean = X.colwise().mean();
    X.rowwise() -= mean.transpose();

    std::cout << "Calculating covariance matrix..." << std::endl;

    // Calcula a matriz de covariância
    Eigen::MatrixXd cov = X.transpose() * X / (n - 1);
    std::cout << "Dimensões da matriz cov: " << cov.rows() << " x " << cov.cols() << std::endl;

    std::cout << "Calculating eigenvalues and eigenvectors..." << std::endl;

    // Calcula os autovalores e autovetores da matriz de covariância
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(cov);
    Eigen::MatrixXd eigen_vectors = eig.eigenvectors().transpose();
    Eigen::VectorXd eigen_values = eig.eigenvalues();

    std::cout << "Calculating principal components..." << std::endl;

    // Seleciona os principais componentes
    Eigen::MatrixXd components = eigen_vectors.topRows(num_components_);

    std::cout << "Applying transformation..." << std::endl;

    // Aplica a transformação linear para reduzir a dimensão dos dados
    Eigen::MatrixXd Y = X * components.transpose();

    std::cout << "Converting output to vector of vectors..." << std::endl;

    // Converte a matriz do Eigen de volta para um vetor de vetores
    std::vector<std::vector<double>> output(n, std::vector<double>(num_components_));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < num_components_; j++) {
            output[i][j] = Y(i, j);
        }
    }

    return output;
}
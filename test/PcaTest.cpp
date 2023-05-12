#include <gtest/gtest.h>

#include "preprocess/Pca.h"

TEST(PCATest, DimensionalityReduction) {
    // Crie um objeto PCA com o número desejado de componentes principais
    Pca pca(2);

    // Defina um caso de teste com dados de entrada
    std::vector<std::vector<double>> input = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

    // Execute o PCA nos dados de entrada
    std::vector<std::vector<double>> output = pca.process(input);

    // Verifique se a dimensionalidade reduzida está correta
    ASSERT_EQ(output.size(), input.size());
    for (size_t i = 0; i < output.size(); i++) {
        ASSERT_EQ(output[i].size(), 2);
    }
}

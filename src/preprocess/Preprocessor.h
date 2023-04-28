#include <vector>

class Preprocessor {
public:
    virtual std::vector<std::vector<double>> process(
        const std::vector<std::vector<double>>& input) = 0;
    virtual ~Preprocessor() {}
};
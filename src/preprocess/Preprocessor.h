#include <vector>

/**
 * This class defines an interface for preprocessing data.
 */
class Preprocessor {
public:
    /**
     * Abstract method to process data.
     *
     * @param input The input data
     * @return std::vector<std::vector<double>> The processed data
     */
    virtual std::vector<std::vector<double>> process(
        const std::vector<std::vector<double>>& input) = 0;

    /**
     * Virtual destructor for the interface
     */
    virtual ~Preprocessor() {}
};

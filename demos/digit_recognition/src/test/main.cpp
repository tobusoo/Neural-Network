#include <fstream>
#include <iostream>

#include <ImageDataSet.hpp>
#include <nn/nn.hpp>

int main(int argc, char* argv[])
{
    srand(time(0));
    if (argc < 2) {
        std::cerr << "Usage: ./test <path to model>\n";
        return 1;
    }

    NN* nn;
    std::cout << "Loading NN model...\n";
    std::ifstream file_nn(argv[1]);
    json model = json::parse(file_nn);
    nn = new NN(model);
    std::cout << "NN model is loaded!\n\n";

    std::cout << "Loading DataSet...\n";
    ImageDataSet dataset;
    dataset.load_images("dataset/test", 10000, 10000);
    std::cout << "DataSet is loaded!\n\n";

    std::cout << "Testing " << argv[1] << "...\n";
    auto dt = dataset.get_batch(0);
    double cost = nn->cost(dt.inputs, dt.outputs);
    std::cout << "Cost: " << cost << '\n';

    return 0;
}
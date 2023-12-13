#include <fstream>
#include <iostream>

#include <ImageDataSet.hpp>
#include <nn/nn.hpp>

#include <sys/time.h>

double wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

int main(int argc, char* argv[])
{
    srand(time(0));

    if (argc < 5) {
        std::cerr << "Usage: ./train <num of train> <num of iter> <max epoch> "
                     "<1(load model) or 0(don't load model)>\n";
        return 1;
    }
    size_t iter = std::atoi(argv[2]);
    size_t max_epoch = std::atoi(argv[3]);
    bool need_load_model = std::atoi(argv[4]);

    NN* nn;
    if (need_load_model) {
        std::cout << "Loading NN model...\n";
        std::string str_nn = "model/nn_" + std::string(argv[1]) + ".json";
        std::ifstream file_nn(str_nn);
        json model = json::parse(file_nn);
        // nn = nn_load(model);
        nn = new NN(model);
        std::cout << "NN model is loaded!\n\n";
    } else {
        size_t layers[] = {28 * 28, 128, 36, 10};
        nn = new NN(layers, ARRAY_LEN(layers));
        nn->rand(-1, 1);
    }

    std::cout << "Loading DataSet...\n";
    ImageDataSet dataset;
    dataset.load_images("dataset/train", 100, 600);
    std::cout << "DataSet is loaded!\n\n";
    size_t batch_size = dataset.get_batch_count();

    std::string out_filename("visualization/output_");
    out_filename += argv[1];
    out_filename += ".data";
    std::ofstream file(out_filename, std::ios::app);

    NN g(nn->get_layers(), nn->get_layers_count());
    double rate = 1;
    double correct;
    size_t j = 0;

    double time = wtime();
    double elapsed_time = 0;
    for (size_t i = iter; i <= max_epoch; i++, j++) {
        if (j == batch_size) {
            j = 0;
        }

        auto dt = dataset.get_batch(j);
        nn->backprop(g, dt.inputs, dt.outputs, rate);
        elapsed_time = wtime() - time;
        double cost = nn->cost(dt.inputs, dt.outputs, correct);
        printf("[%s] %ld: correct = %f %%; cost = %f; elapsed time: %f\n",
               argv[1],
               i,
               correct,
               cost,
               elapsed_time);
        file << i << ' ' << correct << ' ' << cost << ' ' << elapsed_time
             << '\n';
    }

    std::cout << "Saving nn...\n";
    json nn_json = nn->save();

    std::string nn_file = "model/nn_" + std::string(argv[1]) + ".json";
    std::ofstream f(nn_file);

    f << std::setw(4) << nn_json;
    std::cout << "Saved nn\n";

    return 0;
}
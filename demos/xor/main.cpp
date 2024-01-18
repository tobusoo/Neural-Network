#include <iostream>

#include <tobuso/nn.hpp>

int main()
{
    srand(time(0));
    tbs::Matrices in = {
            {1, 2, {0, 0}},
            {1, 2, {0, 1}},
            {1, 2, {1, 0}},
            {1, 2, {1, 1}},
    };
    tbs::Matrices out = {
            {1, 1, {0}},
            {1, 1, {1}},
            {1, 1, {1}},
            {1, 1, {0}},
    };

    size_t input_size = 2;
    tbs::Layers layers = {{2, tbs::SIGMOID}, {1, tbs::SIGMOID}};
    tbs::NN nn(input_size, layers);
    nn.rand(-1, 1);

    tbs::NN gradient(input_size, layers);
    double cost = nn.cost(in, out);
    double rate = 1;

    printf("%d: %lf\n", 0, cost);
    for (size_t i = 1; i <= 10 * 1000; i++) {
        cost = nn.cost(in, out);
        printf("%ld: cost = %lf\n", i, cost);
        nn.backprop(gradient, in, out, rate);
    }

    std::cout << "\nTESTING XOR:\n\n";
    cost = nn.cost(in, out);
    printf("Cost: %lf\n", cost);
    int k = 0;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            nn.forward(in[k++]);

            std::cout << "in: " << i << ' ' << j << " out: ";
            nn.output()->print();
        }
    }
}
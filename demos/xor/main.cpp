#include <iostream>

#include <nn/nn.hpp>

int main()
{
    srand(time(0));
    double xor_in[] = {0, 0, 0, 1, 1, 0, 1, 1};
    double xor_out[] = {0, 1, 1, 0};

    std::vector<Matrix> in;
    for (int i = 0; i < 4; ++i) {
        Matrix mat;
        mat.m = &xor_in[i * 2];
        mat.cols = 2;
        mat.rows = 1;

        in.push_back(mat);
    }

    std::vector<Matrix> out;
    for (int i = 0; i < 4; ++i) {
        Matrix mat;
        mat.m = &xor_out[i];
        mat.cols = 1;
        mat.rows = 1;

        out.push_back(mat);
    }

    size_t layers[] = {2, 2, 1};
    NN nn = NN(layers, ARRAY_LEN(layers));
    nn.rand(-1, 1);

    NN gradient(nn.get_layers(), nn.get_layers_count());

    double cost = nn.cost(in, out);
    printf("%d: %lf\n", 0, cost);
    for (size_t i = 1; i <= 10000; i++) {
        cost = nn.cost(in, out);
        printf("%ld: cost = %lf\n", i, cost);
        nn.backprop(gradient, in, out, 1);
    }

    std::cout << "\nTESTING XOR:\n";
    int k = 0;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            nn.forward(in[k++]);

            std::cout << "in: " << i << ' ' << j
                      << " out: " << nn.get_out().m[0] << '\n';
        }
    }

    return 0;
}
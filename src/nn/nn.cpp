#include <nn.hpp>

namespace tbs {

inline static double derivative(double a, ACT_TYPE act_type)
{
    switch (act_type) {
    case SIGMOID:
        return a * (1 - a); // derivative of sigmoid
    case RELU:
        return a >= 0 ? 1 : 0; // derivative of relu
    default:
        assert(0 && "Unreachable");
    }

    return 0;
}

NN::NN(size_t input_layer, Layers& layers) : input_size(input_layer)
{
    if (input_size == 0) {
        fprintf(stderr, "Size of input layer must be more than 0!");
        exit(EXIT_FAILURE);
    } else if (layers.size() == 0) {
        fprintf(stderr,
                "In addition to the input layer, there should be an out "
                "layer at least!");
        exit(EXIT_FAILURE);
    }

    auto layer = new Layer(input_size, layers[0].size, layers[0].act_type);
    assert(layer && "Could not alloc memory for new Layer");
    l.push_back(layer);

    l.reserve(layers.size());
    for (size_t i = 1; i < layers.size(); ++i) {
        layer = new Layer(
                layers[i - 1].size, layers[i].size, layers[i].act_type);
        assert(layer && "Could not alloc memory for new Layer");
        l.push_back(layer);
    }
}

NN::NN(json& data)
{
    size_t layers_count = data["layers_count"];
    input_size = data["input_size"];
    ACT_TYPE act_type = data["layer_1"]["ACT_TYPE"];
    size_t layer_size = data["layer_1"]["size"];
    l.reserve(layers_count);

    auto layer = new Layer(input_size, layer_size, act_type);
    assert(layer && "Could not alloc memory for new Layer");
    auto v = data["layer_1"]["weights"];
    std::copy(v.begin(), v.end(), layer->w.array);
    v = data["layer_1"]["biases"];
    std::copy(v.begin(), v.end(), layer->b.array);

    l.push_back(layer);

    for (size_t i = 1; i < layers_count; ++i) {
        const auto& str = std::to_string(i + 1);
        const auto& layer_str = "layer_" + str;
        size_t next_layer_size = data[layer_str]["size"];
        act_type = data[layer_str]["ACT_TYPE"];
        layer = new Layer(layer_size, next_layer_size, act_type);
        assert(layer && "Could not alloc memory for new Layer");

        layer_size = next_layer_size;
        v = data[layer_str]["weights"];
        std::copy(v.begin(), v.end(), layer->w.array);
        v = data[layer_str]["biases"];
        std::copy(v.begin(), v.end(), layer->b.array);

        l.push_back(layer);
    }
}

json NN::save()
{
    json data;
    int k = 1;

    data["input_size"] = input_size;
    data["layers_count"] = l.size();
    for (auto& i : l) {
        const auto& str = std::to_string(k++);
        const auto& layer = "layer_" + str;
        double* end = i->w.array + (i->w.cols * i->w.rows);

        std::vector<double> weights(i->w.array, end);
        std::vector<double> biases(i->b.array, i->b.array + i->b.cols);
        data[layer]["weights"] = weights;
        data[layer]["biases"] = biases;
        data[layer]["size"] = i->a.cols;
        data[layer]["ACT_TYPE"] = i->act_type;
    }

    return data;
}

const Matrix* NN::output() const
{
    return &l[l.size() - 1]->a;
}

void NN::print() const noexcept
{
    for (auto& i : l) {
        i->b.print();
        i->w.print();
        printf("\n");
    }
}

void NN::forward(Matrix& input) noexcept
{
    assert(input.cols == input_size);
    l[0]->forward(input);
    for (size_t i = 1; i < l.size(); ++i) {
        l[i]->forward(*l[i - 1]);
    }
}

double NN::cost(Matrices in, Matrices out) noexcept
{
    assert(in.size() == out.size() && "Size of in and out must be equal");
    Matrix* out_layer = &l[l.size() - 1]->a;
    size_t out_layer_size = out_layer->cols;
    size_t n = in.size();
    double cost = 0;

    for (size_t i = 0; i < n; ++i) {
        assert(input_size == in[i].cols);
        assert(out_layer_size == out[i].cols);

        forward(in[i]);
        for (size_t j = 0; j < out_layer_size; ++j) {
            double expected = out[i](0, j);
            double diff = (*out_layer)(0, j) - expected;
            cost += diff * diff;
        }
    }

    return cost / n;
}

void NN::rand(double min, double max) noexcept
{
    for (auto& layer : l) {
        layer->rand(min, max);
    }
}

void NN::fill(double x) noexcept
{
    for (auto& layer : l) {
        layer->fill(x);
    }
}

void NN::backprop(NN& g, Matrices in, Matrices out, double rate)
{
    g.fill(0);
    size_t n = in.size();
    size_t layers_count = l.size();

    for (size_t i = 0; i < n; ++i) {
        forward(in[i]);
        for (auto& layer : g.l) {
            layer->a.fill(0);
        }

        for (size_t j = 0; j < out[i].cols; j++) {
            // (out_j - expect_j)
            (*g.output())(0, j) = (*output())(0, j) - out[i](0, j);
        }

        for (int ll = layers_count - 1; ll >= 0; --ll) {
            for (size_t j = 0; j < l[ll]->a.cols; ++j) {
                double a = l[ll]->a(0, j);      // neuron's activation
                double diff = g.l[ll]->a(0, j); // (t_j - out_j)
                double da = derivative(a, l[ll]->act_type);
                double delta = 2 * da * diff; // delta_j

                g.l[ll]->b(0, j) += delta; // change bias

                size_t l_size = ll == 0 ? input_size : l[ll - 1]->a.cols;
                for (size_t k = 0; k < l_size; ++k) {
                    double aa;
                    double w = l[ll]->w(k, j);
                    if (ll == 0) {
                        aa = in[i].array[k];
                    } else {
                        aa = l[ll - 1]->a(0, k);
                        g.l[ll - 1]->a(0, k) += delta * w;
                    }

                    g.l[ll]->w(k, j) += delta * aa;
                }
            }
        }
    }

    // Averaged gradient
    for (size_t i = 0; i < layers_count; ++i) {
        for (size_t j = 0; j < g.l[i]->w.cols * g.l[i]->w.rows; ++j) {
            g.l[i]->w(0, j) /= n;
            l[i]->w(0, j) -= rate * g.l[i]->w(0, j);
        }

        for (size_t j = 0; j < g.l[i]->b.cols; ++j) {
            g.l[i]->b(0, j) /= n;
            l[i]->b(0, j) -= rate * g.l[i]->b(0, j);
        }
    }
}

NN::~NN()
{
    for (auto& i : l)
        delete i;
}

} // namespace tbs

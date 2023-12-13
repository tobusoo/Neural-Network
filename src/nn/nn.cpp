#include <iostream>
#include <nn/nn.hpp>

#define NN_INPUT(nn) (nn)->a[0]                         //  input matrix
#define NN_OUTPUT(nn) (nn)->a[(nn)->m_layers_count - 1] // output matrix

NN::NN(const size_t* layers, size_t layers_count) : m_layers_count(layers_count)
{
    assert(m_layers = new size_t[layers_count]);
    std::copy(layers, layers + layers_count, m_layers);

    a.push_back(matrix_alloc(1, layers[0]));
    for (size_t i = 1; i < layers_count; i++) {
        w.push_back(matrix_alloc(a[i - 1].cols, layers[i]));
        b.push_back(matrix_alloc(1, layers[i]));
        a.push_back(matrix_alloc(1, layers[i]));
    }
}

NN::NN(json& model)
{
    m_layers_count = model["layers_count"];
    m_layers = new size_t[m_layers_count];

    m_layers[0] = model["input_layer"]["size"];
    m_layers[m_layers_count - 1] = model["output_layer"]["size"];
    for (size_t i = 1; i < m_layers_count - 1; i++) {
        const auto& i_str = std::to_string(i);
        const auto& layers = "hidden_" + i_str;
        m_layers[i] = model[layers]["biasis"]["cols"];
    }

    a.push_back(matrix_alloc(1, m_layers[0]));
    for (size_t i = 1; i < m_layers_count; i++) {
        w.push_back(matrix_alloc(a[i - 1].cols, m_layers[i]));
        b.push_back(matrix_alloc(1, m_layers[i]));
        a.push_back(matrix_alloc(1, m_layers[i]));
    }

    for (size_t i = 0; i < m_layers_count - 1; i++) {
        const auto& i_str = std::to_string(i + 1);
        const auto& layers = "hidden_" + i_str;
        const auto& bb = model[layers]["biasis"]["array"];
        const auto& ww = model[layers]["weights"]["array"];

        std::copy(bb.begin(), bb.end(), b[i].m);
        std::copy(ww.begin(), ww.end(), w[i].m);
    }

    const auto& ww = model["output_layer"]["weights"]["array"];
    const auto& bb = model["output_layer"]["biasis"]["array"];
    std::copy(bb.begin(), bb.end(), b[m_layers_count - 2].m);
    std::copy(ww.begin(), ww.end(), w[m_layers_count - 2].m);
}

NN::~NN()
{
    delete[] m_layers;
}

const size_t* NN::get_layers() const noexcept
{
    return m_layers;
}

size_t NN::get_layers_count() const noexcept
{
    return m_layers_count;
}

const Matrix& NN::get_out() const noexcept
{
    return NN_OUTPUT(this);
}

void NN::rand(double min, double max) noexcept
{
    for (size_t i = 0; i < m_layers_count - 1; ++i) {
        matrix_rand(w[i], min, max);
        matrix_rand(b[i], min, max);
    }
}

void NN::fill(double x) noexcept
{
    for (size_t i = 0; i < m_layers_count - 1; ++i) {
        matrix_fill(w[i], x);
        matrix_fill(b[i], x);
    }
}

void NN::forward(Matrix& input) noexcept
{
    assert(input.cols == a[0].cols);
    std::copy(input.m, input.m + input.cols, a[0].m);

    for (size_t i = 0; i < m_layers_count - 1; ++i) {
        matrix_mult(a[i + 1], a[i], w[i]);
        matrix_sum(a[i + 1], b[i]);
        matrix_activation(a[i + 1]);
    }
}

double NN::cost(MatrixArray in, MatrixArray out) noexcept
{
    assert(in.size() == out.size());
    double cost = 0;
    size_t n = in.size();

    for (size_t i = 0; i < n; ++i) {
        assert(NN_INPUT(this).cols == in[i].cols);
        assert(NN_OUTPUT(this).cols == out[i].cols);

        forward(in[i]);
        for (size_t j = 0; j < NN_OUTPUT(this).cols; ++j) {
            double expected = MATRIX_AT(out[i], 0, j);
            double diff = MATRIX_AT(NN_OUTPUT(this), 0, j) - expected;
            cost += diff * diff;
        }
    }

    return cost / n;
}
double NN::cost(MatrixArray in, MatrixArray out, double& correct) noexcept
{
    assert(in.size() == out.size());

    double cost = 0;
    correct = 0;
    size_t n_data = in.size();

    for (size_t i = 0; i < n_data; ++i) {
        assert(NN_INPUT(this).cols == in[i].cols);
        assert(NN_OUTPUT(this).cols == out[i].cols);

        forward(in[i]);
        size_t max_i = 0;
        double max_value = NN_OUTPUT(this).m[0];
        for (size_t j = 1; j < NN_OUTPUT(this).cols; ++j) {
            if (NN_OUTPUT(this).m[j] > max_value) {
                max_value = NN_OUTPUT(this).m[j];
                max_i = j;
            }
        }

        if (MATRIX_AT(out[i], 0, max_i) == 1) {
            correct++;
        }

        for (size_t j = 0; j < NN_OUTPUT(this).cols; ++j) {
            double expected = MATRIX_AT(out[i], 0, j);
            double diff = MATRIX_AT(NN_OUTPUT(this), 0, j) - expected;
            cost += diff * diff;
        }
    }

    correct = (correct / n_data) * 100;
    return cost /= n_data;
}

void NN::backprop(NN& g, MatrixArray input, MatrixArray output, double rate)
{
    g.fill(0);
    size_t n = input.size();

    for (size_t i = 0; i < n; ++i) {
        forward(input[i]);
        for (size_t j = 0; j < m_layers_count - 1; j++) {
            matrix_fill(g.a[j], 0);
        }

        for (size_t j = 0; j < output[i].cols; j++) {
            MATRIX_AT(NN_OUTPUT(&g), 0, j) = MATRIX_AT(NN_OUTPUT(this), 0, j)
                    - MATRIX_AT(output[i], 0, j); // (out_j - expect_j)
        }

        for (size_t l = m_layers_count - 1; l > 0; --l) {
            for (size_t j = 0; j < a[l].cols; ++j) {
                double a = MATRIX_AT(this->a[l], 0, j); // neuron's activation
                double diff = MATRIX_AT(g.a[l], 0, j);  // (t_j - out_j)
                double da = a * (1 - a);                // derivative of sigmoid
                double delta = 2 * da * diff;           // delta_j

                MATRIX_AT(g.b[l - 1], 0, j) += delta; // change bias

                for (size_t k = 0; k < this->a[l - 1].cols; ++k) {
                    double prev_a
                            = MATRIX_AT(this->a[l - 1], 0, k); // prev activ
                    double w = MATRIX_AT(this->w[l - 1], k, j);

                    MATRIX_AT(g.w[l - 1], k, j) += delta * prev_a;
                    MATRIX_AT(g.a[l - 1], 0, k) += delta * w;
                }
            }
        }
    }

    // Averaged gradient
    for (size_t i = 0; i < g.m_layers_count - 1; i++) {
        for (size_t j = 0; j < g.w[i].rows; ++j) {
            for (size_t k = 0; k < g.w[i].cols; ++k) {
                MATRIX_AT(g.w[i], j, k) /= n;
                MATRIX_AT(this->w[i], j, k) -= rate * MATRIX_AT(g.w[i], j, k);
            }
        }

        for (size_t j = 0; j < g.b[i].cols; ++j) {
            MATRIX_AT(g.b[i], 0, j) /= n;
            MATRIX_AT(this->b[i], 0, j) -= rate * MATRIX_AT(g.b[i], 0, j);
        }
    }
}

json NN::save()
{
    size_t cnt = m_layers_count;
    double* end = b[cnt - 2].m + (b[cnt - 2].cols * b[cnt - 2].rows);
    std::vector<double> v(b[cnt - 2].m, end);

    end = w[cnt - 2].m + (w[cnt - 2].cols * w[cnt - 2].rows);
    std::vector<double> ww(w[cnt - 2].m, end);

    json model;
    model["layers_count"] = cnt;
    model["input_layer"]["size"] = m_layers[0];
    model["output_layer"]["size"] = m_layers[cnt - 1];
    model["output_layer"]["biasis"]["rows"] = b[cnt - 2].rows;
    model["output_layer"]["biasis"]["cols"] = b[cnt - 2].cols;
    model["output_layer"]["biasis"]["array"] = v;
    model["output_layer"]["weights"]["rows"] = w[cnt - 2].rows;
    model["output_layer"]["weights"]["cols"] = w[cnt - 2].cols;
    model["output_layer"]["weights"]["array"] = ww;

    for (size_t i = 1; i < cnt - 1; ++i) {
        end = b[i - 1].m + (b[i - 1].rows * b[i - 1].cols);
        std::vector<double> vb(b[i - 1].m, end);
        end = w[i - 1].m + (w[i - 1].rows * w[i - 1].cols);
        std::vector<double> vw(w[i - 1].m, end);
        const auto& str = std::to_string(i);
        const auto& layers = "hidden_" + str;
        model[layers]["biasis"]["rows"] = b[i - 1].rows;
        model[layers]["biasis"]["cols"] = b[i - 1].cols;
        model[layers]["biasis"]["array"] = vb;
        model[layers]["weights"]["rows"] = w[i - 1].rows;
        model[layers]["weights"]["cols"] = w[i - 1].cols;
        model[layers]["weights"]["array"] = vw;
    }

    return model;
}

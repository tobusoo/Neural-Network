#pragma once

#include <filesystem>
#include <random>
#include <vector>

#include <nn/matrix.h>

namespace fs = std::filesystem;

struct DataBatch {
    using MatrixArray = std::vector<Matrix>;

    MatrixArray inputs;
    MatrixArray outputs;
    size_t n;

    DataBatch();

    void add(Matrix in, Matrix out);
};

class DataSet {
protected:
    std::vector<DataBatch*> v;
    size_t batch_count;
    size_t examples_count;
    std::default_random_engine random_engine;

public:
    DataSet();
    ~DataSet();

    void shuffle();
    size_t get_batch_count() const;
    size_t get_examples_count() const;
    DataBatch get_batch(size_t index);
};
#pragma once

#include <filesystem>
#include <random>
#include <vector>

#include "matrix.hpp"

namespace fs = std::filesystem;
namespace tbs {

struct DataBatch {
    using Matrices = std::vector<tbs::Matrix>;

    Matrices inputs;
    Matrices outputs;
    size_t n;

    DataBatch();

    void add(tbs::Matrix& in, tbs::Matrix& out);
};

// DataSet
//
// This class is useless by default,
// to work with it, you need to inherit from it and
// implement a method to load the dataset.
//
//  A simple example of a derived class can be found here:
//      header: demos/digit_recognition/include/ImageDataSet.hpp
//      src: demos/digit_recognition/src/image_dataset.cpp
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

} // namespace tbs
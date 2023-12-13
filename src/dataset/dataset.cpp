#include <algorithm>
#include <random>

#include <dataset.hpp>

DataBatch::DataBatch() : n(0){};

void DataBatch::add(Matrix in, Matrix out)
{
    inputs.emplace_back(in);
    outputs.emplace_back(out);
    n++;
}

DataSet::DataSet() : batch_count(0), examples_count(0), random_engine()
{
}

DataSet::~DataSet()
{
    for (auto& i : v)
        delete i;
}

void DataSet::shuffle()
{
    std::shuffle(v.begin(), v.end(), random_engine);
}

size_t DataSet::get_batch_count() const
{
    return batch_count;
}

size_t DataSet::get_examples_count() const
{
    return examples_count;
}

DataBatch DataSet::get_batch(size_t index)
{
    return *v.at(index);
}

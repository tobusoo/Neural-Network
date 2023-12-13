#include <dataset.hpp>

class ImageDataSet : public DataSet {
public:
    ImageDataSet();

    void load_images(fs::path path, size_t batch_size, size_t max_bathces);
};
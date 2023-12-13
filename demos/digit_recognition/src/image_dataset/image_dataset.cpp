#include <ImageDataSet.hpp>
#include <proimage.h>

ImageDataSet::ImageDataSet() : DataSet(){};

void ImageDataSet::load_images(
        fs::path path, size_t batch_size, size_t max_batches)
{
    if (max_batches == 0)
        return;
    DataBatch* batch = new DataBatch();
    size_t i = 0; // the counter of batches
    size_t j = 0; // the counter of examples in the batch
    batch_count++;

    for (const auto& file : fs::directory_iterator(path)) {
        if (fs::is_regular_file(file)) {
            Matrix input = process_image(file);
            Matrix output = process_image_name(file.path().filename());

            batch->add(input, output);
            j++;

            if (j == batch_size) {
                examples_count += j;
                j = 0;
                i++;

                v.push_back(batch);
                batch = new DataBatch();
            }
            if (i == max_batches)
                break;
        }
    }

    if (j != 0) {
        i++;
        examples_count += j;
        v.emplace_back(batch);
    }

    batch_count = i;
    shuffle();
}

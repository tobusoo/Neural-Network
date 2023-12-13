#include <filesystem>
#include <iostream>

#include <nn/matrix.h>
#include <proimage.h>

double convert_pixel_to_double(const sf::Uint8* pixels)
{
    double temp = (pixels[0] + pixels[1] + pixels[2]);
    temp /= 255 * 3;
    return temp;
}

double color_to_double(sf::Color color)
{
    double temp = color.r + color.g + color.b;
    temp /= 255 * 3;
    return temp;
}

Matrix process_image(fs::path image_path)
{
    sf::Image image;
    if (!image.loadFromFile(image_path)) {
        std::cerr << "Can't load image:" << image_path << '\n';
        exit(EXIT_FAILURE);
    }

    auto size = image.getSize();
    auto arr = image.getPixelsPtr();

    Matrix input = matrix_alloc(1, size.x * size.y);

    int k = 0;
    int l = 0;
    for (unsigned int i = 0; i < size.x; ++i) {
        for (unsigned int j = 0; j < size.y; ++j) {
            MATRIX_AT(input, 0, l) = convert_pixel_to_double(arr + k);
            k += 4;
            l++;
        }
    }

    return input;
}

Matrix process_image_name(fs::path filename)
{
    Matrix out = matrix_alloc(1, 10);
    matrix_fill(out, 0);
    const char* str = filename.c_str();
    int digit = *(str + 10) - '0';

    MATRIX_AT(out, 0, digit) = 1;

    return out;
}
#pragma once

#include <SFML/Graphics.hpp>
#include <filesystem>
#include <nn/matrix.h>

namespace fs = std::filesystem;

double color_to_double(sf::Color color);
double convert_pixel_to_double(const sf::Uint8* pixels);
Matrix process_image(fs::path image_path);
Matrix process_image_name(fs::path filename);
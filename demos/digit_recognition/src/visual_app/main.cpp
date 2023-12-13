#include <SFML/Graphics.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "imgui-SFML.h"
#include "imgui.h"

#include <nn/nn.hpp>
#include <proimage.h>

namespace fs = std::filesystem;

sf::Color change_color(sf::Color color, bool neighboor = false)
{
    int increment = 70;
    if (neighboor)
        increment /= 2;
    if (color.r + increment < 255)
        color.r += increment;
    else
        color.r = 255;
    if (color.g + increment < 255)
        color.g += increment;
    else
        color.g = 255;
    if (color.b + increment < 255)
        color.b += increment;
    else
        color.b = 255;

    return color;
}

ImVec4 LerpColor(const ImVec4& a, const ImVec4& b, double t)
{
    return ImVec4(
            a.x + t * (b.x - a.x),
            a.y + t * (b.y - a.y),
            a.z + t * (b.z - a.z),
            a.w + t * (b.w - a.w));
}

ImVec4 GetProgressBarColor(double progress)
{
    const ImVec4 red(1.0f, 0.0f, 0.0f, 1.0f);
    const ImVec4 orange(1.0f, 0.5f, 0.0f, 1.0f);
    const ImVec4 yellow(1.0f, 1.0f, 0.0f, 1.0f);
    const ImVec4 green(0.0f, 1.0f, 0.0f, 1.0f);

    if (progress < 0.35f)
        return LerpColor(red, orange, progress / 0.35f);
    if (progress < 0.65f)
        return LerpColor(orange, yellow, (progress - 0.35f) / 0.35f);
    if (progress < 0.85f)
        return LerpColor(yellow, green, (progress - 0.65f) / 0.65f);

    return green;
}

void draw_on_field(
        sf::RectangleShape v[][28],
        sf::Vector2i mouse_pos,
        Matrix input_nn,
        size_t size)
{
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            auto rect = v[i][j].getGlobalBounds();
            if (rect.contains(mouse_pos.x, mouse_pos.y)) {
                auto color = v[i][j].getFillColor();
                color = change_color(color);
                v[i][j].setFillColor(color);
                input_nn.m[j * size + i] = color_to_double(color);

                if (i > 0) {
                    color = v[i - 1][j].getFillColor();
                    color = change_color(color, true);
                    v[i - 1][j].setFillColor(color);
                    input_nn.m[j * size + (i - 1)] = color_to_double(color);
                }
                if (i < size - 1) {
                    color = v[i + 1][j].getFillColor();
                    color = change_color(color, true);
                    v[i + 1][j].setFillColor(color);
                    input_nn.m[j * size + (i + 1)] = color_to_double(color);
                }
                if (j > 0) {
                    color = v[i][j - 1].getFillColor();
                    color = change_color(color, true);
                    v[i][j - 1].setFillColor(color);
                    input_nn.m[(j - 1) * size + i] = color_to_double(color);
                }
                if (j < size - 1) {
                    color = v[i][j + 1].getFillColor();
                    color = change_color(color, true);
                    v[i][j + 1].setFillColor(color);
                    input_nn.m[(j + 1) * size + i] = color_to_double(color);
                }
            }
        }
    }
}

void clean_field(sf::RectangleShape v[][28], Matrix input_nn, size_t size)
{
    matrix_rand(input_nn, 0, 0);
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            v[i][j].setFillColor(sf::Color(0, 0, 0, 255));
        }
    }
}

int main()
{
    const int weight = 1000;
    const int height = 700;
    const int size = 28;
    const int cell_size = 25;

    std::cout << "Loading NN model...\n";
    std::ifstream file_nn("model/nn_0.json");
    json model = json::parse(file_nn);
    NN* nn;
    nn = new NN(model);
    std::cout << "NN model is loaded!\n\n";

    sf::ContextSettings settings;
    settings.antialiasingLevel = 8;
    sf::VideoMode vmode(weight, height);
    sf::RenderWindow window(
            vmode, "Try to draw a digit", sf::Style::Default, settings);
    window.setPosition(sf::Vector2i(0, 0));
    window.setFramerateLimit(144);

    bool t = ImGui::SFML::Init(window);
    std::cout << t << '\n';

    sf::RectangleShape v[size][size];
    Matrix input_nn = matrix_alloc(1, size * size);
    matrix_rand(input_nn, 0, 0);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            sf::RectangleShape temp(sf::Vector2f(cell_size - 1, cell_size - 1));
            temp.setPosition(sf::Vector2f(i * cell_size, j * cell_size));
            temp.setFillColor(sf::Color(0, 0, 0, 255));
            temp.setOutlineThickness(0.1);
            temp.setOutlineColor(sf::Color(255, 255, 255, 125));
            v[i][j] = temp;
        }
    }

    sf::Event event;
    sf::Vector2i mouse_pos;
    sf::Clock deltaClock;

    while (window.isOpen()) {
        while (window.pollEvent(event)) {
            ImGui::SFML::ProcessEvent(window, event);
            switch (event.type) {
            case sf::Event::Closed:
                window.close();
                break;
            case sf::Event::MouseMoved:
                mouse_pos.x = event.mouseMove.x;
                mouse_pos.y = event.mouseMove.y;
                break;
            default:
                break;
            }
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
                window.close();
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::R))
                clean_field(v, input_nn, size);
            if (sf::Mouse::isButtonPressed(sf::Mouse::Left))
                draw_on_field(v, mouse_pos, input_nn, size);
        }

        nn->forward(input_nn);
        Matrix out = nn->get_out();
        double max = out.m[0];
        int max_i = 0;
        ImGui::SFML::Update(window, deltaClock.restart());
        ImGui::Begin("OUTPUT");
        for (size_t i = 0; i < out.cols; i++) {
            ImVec4 originalFillColor
                    = ImGui::GetStyle().Colors[ImGuiCol_PlotHistogram];
            ImGui::GetStyle().Colors[ImGuiCol_PlotHistogram]
                    = GetProgressBarColor(out.m[i]);
            ImGui::ProgressBar(out.m[i], {-1, 0}, std::to_string(i).c_str());
            ImGui::GetStyle().Colors[ImGuiCol_PlotHistogram]
                    = originalFillColor;
            if (out.m[i] > max) {
                max = out.m[i];
                max_i = i;
            }
        }
        ImGui::Text("Probably it's: %d", max_i);
        ImGui::End();

        window.clear();
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                window.draw(v[i][j]);
            }
        }
        ImGui::SFML::Render(window);
        window.display();
    }

    ImGui::SFML::Shutdown();
    return 0;
}
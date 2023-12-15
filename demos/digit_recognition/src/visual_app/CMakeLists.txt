set(target_name visual_app)

add_executable(${target_name})

include(CompileOptions)
set_compile_options(${target_name})

target_sources(
    ${target_name}
    PRIVATE
    main.cpp
)

find_package(SFML 2.5 COMPONENTS graphics system window REQUIRED)

target_link_libraries(
  ${target_name}
  PRIVATE
    tobuso-nn
    sfml-graphics
    sfml-system
    sfml-window
    ImGui-SFML::ImGui-SFML
    proimage
)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/demos/digit_recognition/include)

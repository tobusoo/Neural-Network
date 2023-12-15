set(target_name proimage)

add_library(${target_name} STATIC
  process_image.cpp)

include(CompileOptions)
set_compile_options(${target_name})

target_link_libraries(
  ${target_name}
  PRIVATE
    m
    tobuso-nn
    sfml-graphics
)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/demos/digit_recognition/include)

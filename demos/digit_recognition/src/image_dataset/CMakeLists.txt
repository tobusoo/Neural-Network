set(target_name image_dataset)

add_library(${target_name} STATIC
  image_dataset.cpp)

include(CompileOptions)
set_compile_options(${target_name})

target_link_libraries(
  ${target_name}
  PRIVATE
    m
    tobuso-nn
    proimage
    tobuso-ds
)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/demos/digit_recognition/include)

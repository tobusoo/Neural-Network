set(target_name dataset)

add_library(${target_name} STATIC
  dataset.cpp)

include(CompileOptions)
set_compile_options(${target_name})

target_link_libraries(
  ${target_name}
  PRIVATE
    m
    nn
)

include_directories(${CMAKE_SOURCE_DIR}/include)

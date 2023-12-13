project(nn)

set(target_name nn)

add_library(${target_name} STATIC
  matrix.c
  nn.cpp
)


target_link_libraries(
  ${target_name}
  PRIVATE
    m
    nlohmann_json::nlohmann_json
)

include_directories(${CMAKE_SOURCE_DIR}/include)

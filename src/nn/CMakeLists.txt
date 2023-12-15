project(nn)

set(target_name tobuso-nn)

add_library(${target_name} STATIC
  matrix.c
  nn.cpp
)

target_link_libraries(
  ${target_name}
  PUBLIC
  m
)

set(NN_HEADERS ${CMAKE_SOURCE_DIR}/include/nn/nn.hpp ${CMAKE_SOURCE_DIR}/include/nn/matrix.h)
set_target_properties(${target_name} PROPERTIES PUBLIC_HEADER "${NN_HEADERS}")
target_include_directories( ${target_name} PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/external)

install(TARGETS ${target_name}
          PUBLIC_HEADER DESTINATION include/nn)

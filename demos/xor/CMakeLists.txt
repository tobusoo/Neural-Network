cmake_minimum_required(VERSION 3.22)

project(
    Xor
    VERSION 1.0
    LANGUAGES CXX   
)
set(target_name xor)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

add_executable(${target_name})

target_sources(
    ${target_name}
    PRIVATE
    main.cpp
)

target_link_libraries(
  ${target_name}
  PRIVATE
    tobuso-nn
)

include_directories(${CMAKE_SOURCE_DIR}/include)

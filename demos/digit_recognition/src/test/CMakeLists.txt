set(target_name test)

add_executable(${target_name})

include(CompileOptions)
set_compile_options(${target_name})

target_sources(
    ${target_name}
    PRIVATE
    main.cpp
)

find_library(NN_LIBRARIES NAMES tobuso-nn)

target_link_libraries(
  ${target_name}
  PRIVATE
    tobuso-nn
    image_dataset
)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/demos/digit_recognition/include)

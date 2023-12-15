set(target_name tobuso-ds)

add_library(${target_name} STATIC
  dataset.cpp
)

include(CompileOptions)
set_compile_options(${target_name})

target_link_libraries(
  ${target_name}
  PRIVATE
    m
    tobuso-nn
)

set(DATASET_HEADERS ${CMAKE_SOURCE_DIR}/include/dataset.hpp)
set_target_properties(${target_name} PROPERTIES PUBLIC_HEADER ${DATASET_HEADERS})
target_include_directories( ${target_name} PUBLIC ${CMAKE_SOURCE_DIR}/include)

install(TARGETS ${target_name} 
          PUBLIC_HEADER DESTINATION include)

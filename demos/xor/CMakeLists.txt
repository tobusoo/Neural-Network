set(target_name xor)

add_executable(${target_name})

include(CompileOptions)
set_compile_options(${target_name})

target_sources(
    ${target_name}
    PRIVATE
    main.cpp
)


target_link_libraries(
  ${target_name}
  PRIVATE
    nn
    nlohmann_json::nlohmann_json
)

include_directories(${CMAKE_SOURCE_DIR}/include)

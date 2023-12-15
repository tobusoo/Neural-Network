project(json)

set(target_name json)
add_library(${target_name} INTERFACE)
set(JSON_HEADERS nlohmann/json.hpp)
set_target_properties(${target_name} PROPERTIES PUBLIC_HEADER "nlohmann/json.hpp")

install(TARGETS ${target_name}
          PUBLIC_HEADER DESTINATION include/nlohmann)

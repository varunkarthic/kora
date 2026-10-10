cmake_minimum_required(VERSION 3.21)
project(kora_package_consumer LANGUAGES CXX)

find_package(kora CONFIG REQUIRED)

add_executable(kora_package_consumer main.cpp)
target_link_libraries(kora_package_consumer PRIVATE kora::kora)
target_compile_features(kora_package_consumer PRIVATE cxx_std_20)

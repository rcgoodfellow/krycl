cmake_minimum_required(VERSION 3.0)
project(krycl)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake_modules")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} \
-std=c11 -g -Wall -Wextra -m64 -O3 \
-fdiagnostics-color=always")

include_directories(include/kry)

add_subdirectory(src/core)
add_subdirectory(src/test)
add_subdirectory(kernel)

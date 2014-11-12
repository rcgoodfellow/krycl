cmake_minimum_required(VERSION 3.0)
project(krycl)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11 -g -Wall -Wextra -fdiagnostics-color=always")

add_library(krycl krycl.c)
add_executable(simple_tests simple_tests.c)
target_link_libraries(simple_tests krycl OpenCL)

configure_file(arnoldi.cl arnoldi.cl COPYONLY)

find_package(MKL REQUIRED)

include_directories(${MKL_INCLUDE})

add_library(krytest krytest.c)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MKL_C_FLAGS}")

add_executable(simple_tests simple_tests.c)
target_link_libraries(simple_tests krycl krytest amdocl64 m ${MKL_LIBS})
set_target_properties(simple_tests PROPERTIES LINK_FLAGS "${MKL_LINK_FLAGS}")


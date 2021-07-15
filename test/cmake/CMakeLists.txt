cmake_minimum_required(VERSION 3.5)

project(test_ucc)
find_package(UCC REQUIRED)
add_executable(test_ucc main.cpp)
target_link_libraries(test_ucc ucc::ucc)

cmake_minimum_required(VERSION 3.15)
include(FetchContent)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_COMPILER cc)

project(
  KiFmmDemo
  VERSION 1.0
  LANGUAGES C
  )

  FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.5 # Optionally specify a commit hash, version tag or branch here
)

FetchContent_MakeAvailable(Corrosion)

# Find MPI
find_package(MPI REQUIRED)

corrosion_import_crate(MANIFEST_PATH ../Cargo.toml CRATE_TYPES cdylib PROFILE release FEATURES mpi)

# Single node demo
add_executable(single_node_demo single_node.c)
target_include_directories(single_node_demo PUBLIC ../include)
target_link_libraries(single_node_demo PUBLIC kifmm)


# Multi node demo (+link against MPI)
add_executable(multi_node_demo multi_node.c)
target_include_directories(multi_node_demo PUBLIC ../include)
target_link_libraries(multi_node_demo PUBLIC kifmm MPI::MPI_C)
target_include_directories(multi_node_demo PRIVATE ${MPI_C_INCLUDE_DIRS})

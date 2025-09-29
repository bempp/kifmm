cmake_minimum_required(VERSION 3.15)
include(FetchContent)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_COMPILER cc)

project(
  WeakScalingCrayPAT
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

# --- Option to enable CrayPAT support ---
option(ENABLE_CRAYPAT "Link against CrayPAT (pat_api.h, -lpat)" OFF)

corrosion_import_crate(MANIFEST_PATH ../../../kifmm/Cargo.toml CRATE_TYPES cdylib PROFILE release FEATURES mpi)

# Weak scaling script
add_executable(weak_scaling weak_scaling.c)
target_include_directories(weak_scaling PUBLIC ../../../kifmm/include)
target_link_libraries(weak_scaling PUBLIC kifmm MPI::MPI_C)
target_include_directories(weak_scaling PRIVATE ${MPI_C_INCLUDE_DIRS})

if(ENABLE_CRAYPAT)
  # Find pat_api.h
  find_path(CRAYPAT_INCLUDE_DIRS
    NAMES pat_api.h
    PATHS /opt/cray/pe/perftools/default/include
  )

  if (NOT CRAYPAT_INCLUDE_DIRS)
    message(FATAL_ERROR "CrayPAT requested but not found (pat_api.h)")
  endif()

  target_include_directories(weak_scaling PRIVATE ${CRAYPAT_INCLUDE_DIRS})
  target_link_options(weak_scaling PRIVATE "-Wl,--allow-shlib-undefined")
  message(STATUS "CrayPAT enabled: remember to run 'pat_build' and 'pat_run' on the executable.")
endif()

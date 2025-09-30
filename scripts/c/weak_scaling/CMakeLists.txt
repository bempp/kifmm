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

# --- Option to enable single/double
option(USE_DOUBLE "Use double precision (default: single)" OFF)

if(USE_DOUBLE)
  add_compile_definitions(FMM_USE_DOUBLE)
else()
  add_compile_definitions(FMM_USE_FLOAT)
endif()

# Import Rust crate as static lib
corrosion_import_crate(
  MANIFEST_PATH ../../../kifmm/Cargo.toml
  CRATE_TYPES staticlib
  PROFILE release
  FEATURES mpi
)

# Weak scaling script
add_executable(weak_scaling_fft weak_scaling_fft.c)
target_include_directories(weak_scaling_fft PUBLIC ../../../kifmm/include)
target_link_libraries(weak_scaling_fft PUBLIC kifmm MPI::MPI_C)
target_include_directories(weak_scaling_fft PRIVATE ${MPI_C_INCLUDE_DIRS})
target_include_directories(weak_scaling_fft PRIVATE ../../c) # Helper functions

if(ENABLE_CRAYPAT)
  # Find pat_api.h
  find_path(CRAYPAT_INCLUDE_DIRS
    NAMES pat_api.h
    PATHS /opt/cray/pe/perftools/default/include
  )

  if (NOT CRAYPAT_INCLUDE_DIRS)
    message(FATAL_ERROR "CrayPAT requested but not found (pat_api.h)")
  endif()

  target_include_directories(weak_scaling_fft PRIVATE ${CRAYPAT_INCLUDE_DIRS})
  target_link_options(weak_scaling_fft PRIVATE "-Wl,--allow-shlib-undefined")
  message(STATUS "CrayPAT enabled: remember to run 'pat_build' and 'pat_run' on the executable.")
endif()

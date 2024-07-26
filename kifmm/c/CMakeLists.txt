cmake_minimum_required(VERSION 3.15)
include(FetchContent)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

corrosion_import_crate(MANIFEST_PATH ../Cargo.toml CRATE_TYPES cdylib PROFILE release)

add_executable(fmm_demo main.c)
target_include_directories(fmm_demo PUBLIC ../include)
target_link_libraries(fmm_demo PUBLIC kifmm)

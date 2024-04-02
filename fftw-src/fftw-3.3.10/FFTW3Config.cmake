# defined since 2.8.3
if (CMAKE_VERSION VERSION_LESS 2.8.3)
  get_filename_component (CMAKE_CURRENT_LIST_DIR ${CMAKE_CURRENT_LIST_FILE} PATH)
endif ()

# Allows loading FFTW3 settings from another project
set (FFTW3_CONFIG_FILE "${CMAKE_CURRENT_LIST_FILE}")

set (FFTW3_LIBRARIES fftw3)
set (FFTW3_LIBRARY_DIRS /Users/sri/Code/kifmm/target/debug/build/fftw-src-e5eeb3d281cdbf14/out/lib)
set (FFTW3_INCLUDE_DIRS /Users/sri/Code/kifmm/target/debug/build/fftw-src-e5eeb3d281cdbf14/out/include)

include ("${CMAKE_CURRENT_LIST_DIR}/FFTW3LibraryDepends.cmake")

if (CMAKE_VERSION VERSION_LESS 2.8.3)
  set (CMAKE_CURRENT_LIST_DIR)
endif ()

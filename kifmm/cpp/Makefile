# Compiler
CC = clang++

# Compiler Flags
CFLAGS = -std=c++20

# Directories
BUILD_DIR = build
SOURCE_DIR = src
INCLUDE_DIR = include
RUST_TARGET_DIR = $(abspath ../../target)
RUST_DEBUG_DIR = $(RUST_TARGET_DIR)/debug
RUST_RELEASE_DIR = $(RUST_TARGET_DIR)/release
RUST_INCLUDE_DIR = $(RUST_TARGET_DIR)/
RUST_LIB_DIR = $(RUST_TARGET_DIR)/$(BUILD)

# Library name (without the 'lib' prefix and '.dylib' suffix)
LIB_NAME = kifmm

# Source and Object files
SRC = ${SOURCE_DIR}/main.cpp
OBJ = $(BUILD_DIR)/main.o
EXE = $(BUILD_DIR)/main

# Default build mode
BUILD ?= release

# Print debugging information
$(info RUST_TARGET_DIR: $(RUST_TARGET_DIR))
$(info RUST_INCLUDE_DIR: $(RUST_INCLUDE_DIR))
$(info RUST_LIB_DIR: $(RUST_LIB_DIR))
$(info BUILD: $(BUILD))

# Default rule to build the executable
all: $(EXE)

# Rule to recompile and link the executable
$(EXE): $(OBJ) | $(RUST_INCLUDE_DIR)/kifmm.h
	$(CC) -o $@ $^ -L$(RUST_LIB_DIR) -l$(LIB_NAME)

# Rule to compile the source file
$(OBJ): $(SRC) | $(RUST_INCLUDE_DIR)/kifmm.h
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(RUST_INCLUDE_DIR) -I${INCLUDE_DIR} -c $< -o $@

# Ensure the Rust library and header file are built before compiling C++
$(RUST_INCLUDE_DIR)/kifmm.h:
	cargo build --$(BUILD)

# Clean rule to remove generated files
clean:
	rm -rf $(BUILD_DIR)

# Rule to build the Rust library and C++ executable in debug mode
debug:
	$(MAKE) BUILD=debug

# Rule to build the Rust library and C++ executable in release mode
release:
	$(MAKE) BUILD=release

# Format the source files
SOURCES := $(wildcard src/*.cpp include/*.hxx include/*txx)
format:
	clang-format -i $(SOURCES)
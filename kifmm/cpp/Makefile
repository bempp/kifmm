# Compiler
CC = clang++

# Compiler Flags
CFLAGS = -std=c++11

# Directories
BUILD_DIR = build
TARGET_DIR = $(abspath ../../target)
DEBUG_DIR = $(TARGET_DIR)/debug
RELEASE_DIR = $(TARGET_DIR)/release
INCLUDE_DIR = $(TARGET_DIR)/
LIB_DIR = $(TARGET_DIR)/$(BUILD)

# Library name (without the 'lib' prefix and '.dylib' suffix)
LIB_NAME = kifmm

# Source and Object files
SRC = main.cpp
OBJ = $(BUILD_DIR)/main.o
EXE = $(BUILD_DIR)/main

# Default build mode
BUILD ?= release

# Print debugging information
$(info TARGET_DIR: $(TARGET_DIR))
$(info INCLUDE_DIR: $(INCLUDE_DIR))
$(info LIB_DIR: $(LIB_DIR))
$(info BUILD: $(BUILD))

# Default rule to build the executable
all: $(EXE)

# Rule to recompile and link the executable
$(EXE): $(OBJ) | $(INCLUDE_DIR)/kifmm.h
	$(CC) -o $@ $^ -L$(LIB_DIR) -l$(LIB_NAME)

# Rule to compile the source file
$(OBJ): $(SRC) | $(INCLUDE_DIR)/kifmm.h
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Ensure the Rust library and header file are built before compiling C++
$(INCLUDE_DIR)/kifmm.h:
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

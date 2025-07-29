cmake_minimum_required(VERSION 3.10)
project(MyProject)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/../inc)

# Source files
set(ALL_SOURCE_FILES
    "${CMAKE_SOURCE_DIR}/../src/decision_tree_classifier.cpp"
    "${CMAKE_SOURCE_DIR}/../src/file_operations.cpp"
    "${CMAKE_SOURCE_DIR}/../src/validation.cpp"
    "${CMAKE_SOURCE_DIR}/src/train_test_split.cpp"
    "${CMAKE_SOURCE_DIR}/src/proposed.cpp"
    "${CMAKE_SOURCE_DIR}/src/main.cpp"
)

# Define configurable parameters with cache
set(DTC_MIN_SAMPLES_SPLIT 10 CACHE STRING "Set minimum number of samples in a node to be split")
set(DTC_MAX_PURITY 0.95 CACHE STRING "Set maximum purity of nodes to be split")
set(PROPOSED_LEVEL 0.95 CACHE STRING "Set level of proposed hierarchical RNN")

# Add executable
add_executable(main ${ALL_SOURCE_FILES})

# Link the shared library to the main executable
target_link_libraries(main PRIVATE)

# Add compile definitions for main target
target_compile_definitions(main PRIVATE 
    DTC_MIN_SAMPLES_SPLIT=${DTC_MIN_SAMPLES_SPLIT}
    DTC_MAX_PURITY=${DTC_MAX_PURITY}

    PROPOSED_LEVEL=${PROPOSED_LEVEL}
)

target_compile_options(main PRIVATE -O3)

cmake_minimum_required(VERSION 3.10)
project(MyProject)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add executable
add_executable(main
    "${CMAKE_SOURCE_DIR}/../../src/decision_tree_classifier.cpp"
    "${CMAKE_SOURCE_DIR}/../../src/file_operations.cpp"
    "${CMAKE_SOURCE_DIR}/../../src/validation.cpp"
    "${CMAKE_SOURCE_DIR}/src/entropy_based_undersampling_approach.cpp"
    "${CMAKE_SOURCE_DIR}/src/main.cpp"
)

# Include directories
target_include_directories(main PRIVATE
    ${CMAKE_SOURCE_DIR}/../../inc   # public headers
    ${CMAKE_SOURCE_DIR}/../inc      # private headers
)

# Add compile defines
target_compile_definitions(main PRIVATE 
    DTC_MIN_SAMPLES_SPLIT=${DTC_MIN_SAMPLES_SPLIT}  # Set minimum number of samples in a node to be split
    DTC_MAX_PURITY=${DTC_MAX_PURITY}                # Set maximum purity of nodes to be split 
)

# Add optimization flags
target_compile_options(main PRIVATE -O3)
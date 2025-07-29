#ifndef FILE_OPERATIONS_H
#define FILE_OPERATIONS_H

#include <vector>
#include <limits>  // std::numeric_limits<T>::max();
#include <fstream> // std::ifstream
#include <sstream> // std::stringstream
#include <iostream>

typedef struct Dataset{
    uint32_t n_classes;
    std::vector<std::vector<float>> training_set;
    std::vector<std::vector<float>> testing_set;
}Dataset;

// The labels in the training and testing sets must start from 1 and be placed after the attributes
// Return the normalized training and testing sets
Dataset ReadTrainingAndTestingSet(std::string training_path, std::string testing_path);

#endif // FILE_OPERATIONS_H
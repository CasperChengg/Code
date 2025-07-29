#ifndef TRAIN_TEST_SPLIT_H
#define TRAIN_TEST_SPLIT_H

#include <random>    // std::default_random_engine
#include <chrono>    // std::chrono  
#include <algorithm> // shuffle
#include <iostream>
#include "../../inc/file_operations.h"

void TrainTestSplit(const std::vector<std::vector<float>>&dataset, const float split_ratio, std::vector<std::vector<float>> &training_set, std::vector<std::vector<float>> &testing_set, const uint32_t n_classes);
void KFoldSplit(const std::vector<std::vector<float>>& dataset, const uint32_t n_classes, const uint32_t k, std::vector<std::vector<std::vector<float>>> &training_set, std::vector<std::vector<std::vector<float>>> &testing_set);

#endif

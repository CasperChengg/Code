#ifndef RANDOM_UNDER_SAMPLING_H
#define RANDOM_UNDER_SAMPLING_H

#include <cstdint>      // uint32_t
#include <vector>       // std::vector
#include <memory>       // std::unique_ptr
#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono  
#include <algorithm>    // shuffle
#include <iostream>     // std::cerr, std::endl

class RandomUnderSampler
{
    public:
        RandomUnderSampler(){};
        ~RandomUnderSampler() = default;
        std::vector<std::vector<float>> fit_resample(const std::vector<std::vector<float>> &tra_set, const uint32_t n_classes);
};

#endif

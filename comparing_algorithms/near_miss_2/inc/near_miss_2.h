#ifndef RANDOM_UNDER_SAMPLING_H
#define RANDOM_UNDER_SAMPLING_H

#include <cmath>
#include <queue>
#include <vector>
#include <utility>
#include <cstdint>
#include <iostream>
#include <algorithm>

class NearMiss2{
    public:
        NearMiss2(const uint32_t k = 3) :k_(k){};
        ~NearMiss2() = default;
        std::vector<std::vector<float>> fit_resample(const std::vector<std::vector<float>> &tra_set, const uint32_t n_classes);
    private:
        const uint32_t k_;
};  

#endif

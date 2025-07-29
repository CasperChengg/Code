#ifndef INSTANCE_HARDNESS_THRESHOLD_H
#define INSTANCE_HARDNESS_THRESHOLD_H

#include <vector>    // std::vector
#include <utility>   // std::pair
#include <random>    // std::default_random_engine
#include <chrono>    // std::chrono  
#include <algorithm> // shuffle
#include <iostream>
#include "../../../inc/decision_tree_classifier.h"

class InstanceHardnessThreshold
{
    public:
        InstanceHardnessThreshold(const struct decision_tree_parameter &dtc_params, const uint32_t folds = 5) :folds_(folds), dtc_params_(dtc_params){}
        ~InstanceHardnessThreshold() = default;
        std::vector<std::vector<float>> fit_resample(const std::vector<std::vector<float>> &tra_set, const uint32_t n_classes);
    private:
        const uint16_t folds_;
        const struct decision_tree_parameter &dtc_params_;
};

#endif

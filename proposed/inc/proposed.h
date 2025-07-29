#ifndef PROPOSED_H
#define PROPOSED_H

// #define DEBUG_MODE

#ifdef DEBUG_MODE
    #define PDEBUG(fmt, ...) printf(fmt, ##__VA_ARGS__)
    #define DEBUG_BLOCK(fmt) fmt
#else
    #define PDEBUG(fmt, ...)
    #define DEBUG_BLOCK(fmt)
#endif

#include <iomanip> // std::fixed, std::setprecision

#include <cmath>
#include <random>
#include <ctime> // timespec, clock_gettime
#include <queue>
#include <vector>
#include <utility> // std::pair
#include <iostream>
#include <algorithm>
#include "../../inc/validation.h"
#include "../../inc/file_operations.h"
#include "../inc/train_test_split.h"

class Proposed{
    public:
        Proposed(const decision_tree_parameter &dtc_params) :dtc_params_(dtc_params)
        {
            n_classes_ = 0;
        };
        ~Proposed() = default;
        std::vector<std::vector<float>> fit_resample(const std::vector<std::vector<float>> &tra_set, const uint32_t n_classes);
    
    private:
        uint32_t n_classes_;
        uint32_t label_idx_; 
        std::vector<uint32_t> class_cnts_;  // class counts
        std::vector<uint32_t> k_max_;       // adaptive k_max for each class

        const decision_tree_parameter &dtc_params_;
        std::unique_ptr<std::vector<std::vector<float>>> res_set_; // resampled set
        std::unique_ptr<std::vector<std::vector<std::pair<uint32_t, float>>>> dist_mat_;

        std::vector<std::vector<uint32_t>> RNN;
        std::vector<float> inf_scores_;

        void compute_kmax(void);
        void find_RNN(void);
        void compute_inf_scores(const std::vector<std::vector<uint32_t>> &confusion_matrix);
        void rw_select_by_inf_scores(std::vector<bool> &selection_result, const uint32_t n_rounds);
};

#endif


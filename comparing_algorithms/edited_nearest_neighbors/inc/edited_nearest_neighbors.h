#ifndef EDIT_NEAREST_NEIGHBORS_H
#define EDIT_NEAREST_NEIGHBORS_H

#include <cmath>        // sqrt
#include <memory>       // std::unique_ptr
#include <vector>       // std::vector
#include <cstdint>      // uint32_t
#include <utility>      // std::pair
#include <iostream>     // std::cerr, std::endl
#include <algorithm>    // std::max_element, std::distance, std::numeric_limits


class EditedNearestNeighbors{
    public:
        EditedNearestNeighbors(const uint32_t k = 3) : k_(k) 
        {
            n_classes_ = 0;
            res_set_   = nullptr;
            dist_mat_  = nullptr;
        }
        ~EditedNearestNeighbors() = default; // unique_ptr will handle memory cleanup
        std::vector<std::vector<float>> fit_resample(const std::vector<std::vector<float>> &tra_set, const uint32_t n_classes);

    private:
        const uint32_t k_;
        uint32_t n_classes_;
        uint32_t label_idx_;
        std::unique_ptr<std::vector<std::vector<float>>> res_set_;
        std::unique_ptr<std::vector<std::vector<std::pair<uint32_t, float>>>> dist_mat_;
        bool is_noise(const uint32_t src_idx);
};

#endif
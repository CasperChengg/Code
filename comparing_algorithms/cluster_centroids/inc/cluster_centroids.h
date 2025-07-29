#include<vector>
#include<algorithm>
#include<cstdint>
#include<iostream>
#include "./k_means_pp.h"

class ClusterCentroids{
    public:
        ClusterCentroids(const uint32_t max_iters, const float tolerance) :max_iters_(max_iters), tolerance_(tolerance){}
        ~ClusterCentroids() = default;
        std::vector<std::vector<float>> fit_resample(std::vector<std::vector<float>> &tra_set, const uint32_t n_classes);
    
    private:
        const uint32_t max_iters_;
        const float tolerance_;
};
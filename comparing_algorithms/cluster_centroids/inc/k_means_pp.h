#include <cmath>    // sqrt
#include <vector>
#include <limits>   // std::numeric_limits
#include <memory>   // std::unique_ptr
#include <random>   // std::random_device, std::mt19937 gen(), std::uniform_real_distribution<>;
#include <limits>   // std::numeric_limits<float>::max();
#include<iostream>

class KMeansPP
{
    public:
        KMeansPP(const uint32_t max_iter, const float tolerance) : max_iter_(max_iter), tolerance_(tolerance) {}
        ~KMeansPP()
        {
            centroids_.clear();
        };

        void fit(const std::vector<std::vector<float>> &tra_set, const uint32_t n_clusters = 1);
        std::vector<std::vector<float>> get_centroids()
        { 
            return centroids_; 
        }
    
    private:
        const uint32_t max_iter_;
        const float tolerance_;
        std::unique_ptr<std::vector<std::vector<float>>> res_set_;
        std::vector<std::vector<float>> centroids_;  
        void gen_init_centroids(uint32_t n_clusters);
        uint32_t rw_selection(std::vector<float> fitnesses);
};

#ifndef RANDOM_UNDER_SAMPLING_H
#define RANDOM_UNDER_SAMPLING_H

#include <vector>    // std::vector
#include <memory>    // std::unique_ptr
#include <random>    // std::default_random_engine
#include <chrono>    // std::chrono  
#include <algorithm> // shuffle
#include <iostream>

class EntropyBasedUndersampling
{
    public:
        EntropyBasedUndersampling(const uint32_t k = 5) : k_(k)
        {
            n_classes_ = 0;
        };
        ~EntropyBasedUndersampling() = default;
        std::vector<std::vector<float>> fit_resample(const std::vector<std::vector<float>> &tra_set, const uint32_t n_classes);
    
    private:
        const uint32_t k_;
        uint32_t label_idx_;
        uint32_t n_classes_;
        std::vector<uint32_t> class_cnts_;
        
        std::vector<float> lambda_; 
        std::vector<float> cla_lambda_sum_;         // sum of lambda for each class
        std::vector<float> lambda_entro_;           // lambda * log(lambda)
        std::vector<float> cla_lambda_entro_sum_;   // sum of lambda_entro for each class
        
        std::vector<float> pi_; 
        std::vector<float> eta_;
        
        std::vector<float> gamma_;
        std::vector<float> theta_;
        std::unique_ptr<std::vector<std::vector<float>>> res_set_; // resampled set
        void compute_instance_wise_stc(std::vector<std::vector<uint32_t>> &intra_class_nns);
        void compute_class_wise_stc();
        void compute_instance_wise_diff();
        void compute_class_wise_diff();
};

#endif

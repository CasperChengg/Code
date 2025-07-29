#include "../inc/entropy_based_undersampling_approach.h"

inline float euclidean_dist(const std::vector<float> &src, const std::vector<float> &dst)
{
    float square_distance = 0;

    // the last column of vector stores the label
    for(uint32_t feature_idx = 0; feature_idx < src.size() - 1; feature_idx++){
        float diff = src[feature_idx] - dst[feature_idx];
        square_distance += diff * diff;
    }

    return sqrt(square_distance);
}

void EntropyBasedUndersampling::compute_class_wise_diff(void)
{   
    eta_.resize(n_classes_ + 1, 0.f);
    for(uint32_t data_idx = 0; data_idx < res_set_->size(); data_idx++){
        uint32_t label = (*res_set_)[data_idx][label_idx_];
        eta_[label] += pi_[data_idx];
    }

    for(uint32_t class_idx = 1; class_idx <= n_classes_; class_idx++){
        eta_[class_idx] /= class_cnts_[class_idx];
    }
}

void EntropyBasedUndersampling::compute_class_wise_stc()
{
    theta_.resize(n_classes_ + 1, 0.f);

    cla_lambda_sum_.resize(n_classes_ + 1, 0.f);
    lambda_entro_.resize(res_set_->size(), 0.f);
    cla_lambda_entro_sum_.resize(n_classes_ + 1, 0.f);
    for(uint32_t data_idx = 0; data_idx < res_set_->size(); data_idx++){
        uint32_t label = (*res_set_)[data_idx][label_idx_];
        cla_lambda_sum_[label] += lambda_[data_idx];
        if(lambda_[data_idx] > 0.f){
            lambda_entro_[data_idx] = lambda_[data_idx] * log(lambda_[data_idx]);
            cla_lambda_entro_sum_[label] += lambda_entro_[data_idx];
        }  
    }
    
    for(uint32_t class_idx = 1; class_idx <= n_classes_; class_idx++){
        theta_[class_idx] = -1.f * cla_lambda_entro_sum_[class_idx] / cla_lambda_sum_[class_idx] + log(cla_lambda_sum_[class_idx]);
        theta_[class_idx] /= class_cnts_[class_idx];
    }
}
																
void EntropyBasedUndersampling::compute_instance_wise_stc(std::vector<std::vector<uint32_t>> &intra_class_nns)
{    
    std::vector<std::vector<std::pair<uint32_t, float>>> dist_mat(res_set_->size(), std::vector<std::pair<uint32_t, float>>(res_set_->size(), {0, 0.f}));
    for(uint32_t src_idx = 0; src_idx < res_set_->size(); src_idx++){
        dist_mat[src_idx][src_idx] = {src_idx, std::numeric_limits<float>::max()};
        for(uint32_t dst_idx = src_idx + 1; dst_idx < res_set_->size(); dst_idx++){
            float dist = euclidean_dist((*res_set_)[src_idx], (*res_set_)[dst_idx]);
            dist_mat[src_idx][dst_idx] = {dst_idx, dist};
            dist_mat[dst_idx][src_idx] = {src_idx, dist};
        }
    }
    
    lambda_.resize(res_set_->size(), 0.f);
    
    intra_class_nns.clear();
    intra_class_nns.resize(res_set_->size());
    for(uint32_t src_idx = 0; src_idx < res_set_->size(); src_idx++){
        uint32_t src_label = (*res_set_)[src_idx][label_idx_];

        std::partial_sort(dist_mat[src_idx].begin(), dist_mat[src_idx].begin() + k_, dist_mat[src_idx].end(), 
            [](const std::pair<uint32_t, float> &a, const std::pair<uint32_t, float> &b){return a.second < b.second;});

        for(uint32_t k = 0; k < k_; k++){
            uint32_t nn_idx   = dist_mat[src_idx][k].first;
            uint32_t nn_label = (*res_set_)[nn_idx][label_idx_];

            if(nn_label == src_label){
                intra_class_nns[src_idx].emplace_back(nn_idx);
                if(dist_mat[src_idx][k].second > 0.f){
                    lambda_[src_idx] += (1.f / dist_mat[src_idx][k].second);
                }
            }
        }

        if(intra_class_nns[src_idx].size() > 0){
            lambda_[src_idx] /= intra_class_nns[src_idx].size();
        }
    }
}

void EntropyBasedUndersampling::compute_instance_wise_diff(void)
{
    std::vector<std::vector<uint32_t>> intra_class_nns;
    std::vector<float> class_lambda_sum;

    compute_instance_wise_stc(intra_class_nns); // statistic (stc)
    compute_class_wise_stc();

    std::vector<float> delta(res_set_->size(), 0.f);
    for(uint32_t data_idx = 0; data_idx < res_set_->size(); data_idx++){
        // L_i is the set including the instance i and its intra-class nearest neighbors
        float l_i_lambda_sum       = lambda_[data_idx]; 
        float l_i_lambda_entro_sum = lambda_entro_[data_idx]; 
        
        for(uint32_t k = 0; k < intra_class_nns[data_idx].size(); k++){
            uint32_t intra_class_nn_idx = intra_class_nns[data_idx][k];
            l_i_lambda_sum       += lambda_[intra_class_nn_idx];
            l_i_lambda_entro_sum += lambda_entro_[intra_class_nn_idx];
        }

        uint32_t label = (*res_set_)[data_idx][label_idx_];
        float cla_lambda_sum_i       = cla_lambda_sum_[label] - l_i_lambda_sum; // exclude the instance i and its intra-class nearest neighbors
        float cla_lambda_entro_sum_i = cla_lambda_entro_sum_[label] - l_i_lambda_entro_sum;
        float theta_i = -1.f * cla_lambda_entro_sum_i / cla_lambda_sum_i + log(cla_lambda_sum_i);
        if(theta_i > 0.f){
            theta_i /= (class_cnts_[label] - intra_class_nns[data_idx].size());
            delta[data_idx] = theta_[label] * log(theta_[label] / theta_i);
        }
    }
    
    float exp_sum = 0.f;
    pi_.resize(res_set_->size(), 0.f);
    for(uint32_t data_idx = 0; data_idx < res_set_->size(); data_idx++){
        pi_[data_idx] = exp(delta[data_idx]);
        exp_sum += pi_[data_idx];
    }
    
    for(uint32_t data_idx = 0; data_idx < res_set_->size(); data_idx++){
        pi_[data_idx] /= exp_sum;
    }
}

std::vector<std::vector<float>> EntropyBasedUndersampling::fit_resample(const std::vector<std::vector<float>> &tra_set, const uint32_t n_classes)
{
    this->n_classes_ = n_classes;
    this->label_idx_ = tra_set[0].size() - 1;
    res_set_ = std::make_unique<std::vector<std::vector<float>>>(tra_set);

    class_cnts_.resize(n_classes_ + 1, 0);
    for(uint32_t data_idx = 0; data_idx < res_set_->size(); data_idx++){
        uint32_t label = (*res_set_)[data_idx][label_idx_];
        class_cnts_[label]++;
    }

    compute_instance_wise_diff();
    compute_class_wise_diff();

    float max_eta = *std::max_element(eta_.begin() + 1, eta_.end());
    for(uint32_t class_idx = 1; class_idx <= n_classes_; class_idx++){
        float delta = max_eta - eta_[class_idx];
        while(delta > 0.f && class_cnts_[class_idx] > 1){
            int min_idx_in_class = -1;
            for(uint32_t data_idx = 0; data_idx < res_set_->size(); data_idx++){
                uint32_t label = (*res_set_)[data_idx][label_idx_];
                if(label == class_idx){
                    if(pi_[data_idx] < pi_[min_idx_in_class] || min_idx_in_class == -1){
                        min_idx_in_class = data_idx;
                    }
                }
            }
            class_cnts_[class_idx]--;
            res_set_->erase(res_set_->begin() + min_idx_in_class);
            compute_instance_wise_diff();
            compute_class_wise_diff();
            delta = max_eta - eta_[class_idx];
        }
    }

    class_cnts_.assign(n_classes + 1, 0); // reset class counts
    for(uint32_t data_idx = 0; data_idx < (*res_set_).size(); data_idx++){   
        uint32_t label = (*res_set_)[data_idx][label_idx_];
        class_cnts_[label]++;
    }

    return *res_set_;
}
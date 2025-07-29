#include "../inc/edited_nearest_neighbors.h"

static inline float euclidean_dist(const std::vector<float> &src, const std::vector<float> &dst)
{
    float square_distance = 0;
    
    // last column stores label
    for(uint32_t feature_idx = 0; feature_idx < src.size() - 1; feature_idx++){
        float diff = src[feature_idx] - dst[feature_idx];
        square_distance += diff * diff;
    }
    return sqrt(square_distance);
}

bool EditedNearestNeighbors::is_noise(const uint32_t src_idx)
{
    std::partial_sort((*dist_mat_)[src_idx].begin(), (*dist_mat_)[src_idx].begin() + k_, (*dist_mat_)[src_idx].end(), 
                        [](const std::pair<uint32_t, float> &a, const std::pair<uint32_t, float> &b)
                            {return a.second < b.second;}); // acessing the k nearest neighbors
    
    std::vector<uint32_t> local_class_cnts(n_classes_ + 1, 0);
    for(uint32_t k = 0; k < 3; k++){
        uint32_t nn_idx   = (*dist_mat_)[src_idx][k].first;
        uint32_t nn_label = (*res_set_)[nn_idx][label_idx_];
        local_class_cnts[nn_label]++;
    }

    auto max_it = std::max_element(local_class_cnts.begin() + 1, local_class_cnts.end());
    uint32_t local_maj_label = std::distance(local_class_cnts.begin(), max_it);

    uint32_t src_label = (*res_set_)[src_idx][label_idx_];
    if(src_label != local_maj_label){
        return true;
    }
    else{
        return false;
    }
}



std::vector<std::vector<float>> EditedNearestNeighbors::fit_resample(const std::vector<std::vector<float>> &tra_set, const uint32_t n_classes)
{
    this->label_idx_ = tra_set[0].size() - 1; // assuming last column is label
    this->n_classes_ = n_classes;
    res_set_ = std::make_unique<std::vector<std::vector<float>>>(tra_set); // resampled_set
     
    std::vector<uint32_t> class_cnts(n_classes + 1, 0); 
    for(uint32_t data_idx = 0; data_idx < (*res_set_).size(); data_idx++){   
        uint32_t label = (*res_set_)[data_idx][label_idx_];
        class_cnts[label]++;
    }

    this->dist_mat_ = std::make_unique<std::vector<std::vector<std::pair<uint32_t, float>>>>(
        res_set_->size(), 
        std::vector<std::pair<uint32_t, float>>(res_set_->size(), {0, 0.f})
    );

    for(uint32_t src_idx = 0; src_idx < (*res_set_).size(); src_idx++){
        (*dist_mat_)[src_idx][src_idx] = {src_idx, std::numeric_limits<float>::max()};
        for(uint32_t dst_idx = src_idx + 1; dst_idx < (*res_set_).size(); dst_idx++){
            float distance = euclidean_dist((*res_set_)[src_idx], (*res_set_)[dst_idx]);
            (*dist_mat_)[src_idx][dst_idx] = {dst_idx, distance};
            (*dist_mat_)[dst_idx][src_idx] = {src_idx, distance};
        }
    }

    auto min_it = std::min_element(class_cnts.begin() + 1, class_cnts.end());
    uint32_t minor_class_idx = std::distance(class_cnts.begin(), min_it);

    std::vector<bool> is_removed((*res_set_).size(), false);
    for(uint32_t data_idx = 0; data_idx < (*res_set_).size(); data_idx++){
        uint32_t label = (*res_set_)[data_idx][label_idx_];
        if(label != minor_class_idx && is_noise(data_idx)){
            is_removed[data_idx] = true;
        }
    }

    for(int data_idx = (*res_set_).size() - 1; data_idx >= 0; data_idx--){
        if(is_removed[data_idx]){
            (*res_set_).erase((*res_set_).begin() + data_idx);
        }
    }

    return *res_set_;
}
#include "../inc/near_miss_2.h"

static float inline EuclideanDistance(const std::vector<float> &src, const std::vector<float> &dst)
{
    float square_distance = 0;
    
    // last column stores label
    for(uint32_t feature_idx = 0; feature_idx < src.size() - 1; feature_idx++){
        float diff = src[feature_idx] - dst[feature_idx];
        square_distance += diff * diff;
    }

    return sqrt(square_distance);
}

std::vector<std::vector<float>> NearMiss2::fit_resample(const std::vector<std::vector<float>> &tra_set, const uint32_t n_classes)
{
    const uint32_t label_idx = tra_set[0].size() - 1;
    std::vector<std::vector<float>> res_set = tra_set; // resampled set

    std::vector<std::vector<float>> dist_mat(res_set.size(), std::vector<float>(res_set.size(), -1.f));
    for(uint32_t src_idx = 0; src_idx < res_set.size(); src_idx++){
        dist_mat[src_idx][src_idx] = 0.f;
        for(uint32_t dst_idx = src_idx + 1; dst_idx < res_set.size(); dst_idx++){
            float distance = EuclideanDistance(res_set[src_idx], res_set[dst_idx]);
            dist_mat[src_idx][dst_idx] = distance;
            dist_mat[dst_idx][src_idx] = distance;
        }
    }

    std::vector<uint32_t> class_cnts(n_classes + 1, 0);
    for(uint32_t data_idx = 0; data_idx < res_set.size(); data_idx++){
        uint32_t label = res_set[data_idx][label_idx];
        class_cnts[label]++;
    }

    std::vector<std::vector<std::pair<uint32_t, float>>> dist_to_other_classes(n_classes + 1);
    for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++){
        dist_to_other_classes[class_idx].reserve(class_cnts[class_idx]);
    }

    for(uint32_t src_idx = 0; src_idx < res_set.size(); src_idx++){
        uint32_t src_label = res_set[src_idx][label_idx];
        std::vector<float> neighbors;
        neighbors.reserve(res_set.size() - class_cnts[src_label]);

        for(uint32_t dst_idx = 0; dst_idx < res_set.size(); dst_idx++){
            uint32_t dst_label = res_set[dst_idx][label_idx];
            if(dst_label != src_label){
                neighbors.emplace_back(dist_mat[dst_idx][src_idx]);
            }
        }

        std::partial_sort(neighbors.begin(), neighbors.begin() + k_, neighbors.end(), 
            [](const float &a, const float &b){return a > b;}); // sort in descending order

        float sum_dist = 0.f;
        for(uint32_t k = 0; k < 3; k++){
            sum_dist += neighbors[k];
        }
    
        dist_to_other_classes[src_label].emplace_back(src_idx, sum_dist / 3.f);
    }

    std::vector<bool> is_preserved(res_set.size(), false);
    const uint32_t num_data_to_preserve = *std::min_element(class_cnts.begin() + 1, class_cnts.end()); // per class
    for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++){
        std::partial_sort(dist_to_other_classes[class_idx].begin(), dist_to_other_classes[class_idx].begin() + num_data_to_preserve, dist_to_other_classes[class_idx].end(), 
            [](const std::pair<uint32_t, float> &a, std::pair<uint32_t, float> &b){return a.second < b.second;}); // sort in acscending order
        
            for(uint32_t idx = 0; idx < num_data_to_preserve; idx++){
            uint32_t data_idx = dist_to_other_classes[class_idx][idx].first;
            is_preserved[data_idx] = true;
        }
    }

    for(int data_idx = res_set.size() - 1; data_idx >= 0; data_idx--){
        if(!is_preserved[data_idx]){
            res_set.erase(res_set.begin() + data_idx);
        }
    }

    class_cnts.assign(n_classes + 1, 0); // reset class counts
    for(uint32_t data_idx = 0; data_idx < res_set.size(); data_idx++){   
        uint32_t label = res_set[data_idx][label_idx];
        class_cnts[label]++;
    }

    return res_set;
}
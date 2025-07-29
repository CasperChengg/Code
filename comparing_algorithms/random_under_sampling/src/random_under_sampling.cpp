#include "../inc/random_under_sampling.h"

std::vector<std::vector<float>> RandomUnderSampler::fit_resample(const std::vector<std::vector<float>> &tra_set, const uint32_t n_classes)
{
    const uint32_t label_idx = tra_set[0].size() - 1;
    std::vector<std::vector<float>> res_set = tra_set;
    
    std::vector<uint32_t> class_cnts(n_classes + 1, 0); 
    for(uint32_t data_idx = 0; data_idx < res_set.size(); data_idx++){   
        uint32_t label = res_set[data_idx][label_idx];
        class_cnts[label]++;
    }
    
    std::vector<std::vector<uint32_t>> data_idxes_by_class(n_classes + 1);
    for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++){
        data_idxes_by_class[class_idx].reserve(class_cnts[class_idx]);
    }

    for(uint32_t data_idx = 0; data_idx < res_set.size(); data_idx++){
        uint32_t label = res_set[data_idx][label_idx];
        data_idxes_by_class[label].emplace_back(data_idx);
    }

    uint32_t num_data_to_preserve = *min_element(class_cnts.begin() + 1, class_cnts.end());

    std::vector<bool> is_preserved(res_set.size(), false);
    for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++){
        uint32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
        shuffle(data_idxes_by_class[class_idx].begin(), data_idxes_by_class[class_idx].end(), 
                    std::default_random_engine(seed));

        for(uint32_t idx = 0; idx < num_data_to_preserve; idx++){
            uint32_t data_idx = data_idxes_by_class[class_idx][idx];    
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
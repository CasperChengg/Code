#include "../inc/instance_hardness_threshold.h"

std::vector<std::vector<float>> InstanceHardnessThreshold::fit_resample(const std::vector<std::vector<float>> &tra_set, const uint32_t n_classes)
{
    const uint32_t label_idx = tra_set[0].size() - 1;
    std::vector<std::vector<float>> res_set = tra_set; // resampled set
    std::shuffle(res_set.begin(), res_set.end(), std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count()));

    uint32_t left = 0, right = 0;
    std::vector<std::vector<float>> sub_tra_set;
    std::vector<float> instance_hardnesses(res_set.size(), 0.f);
    for(uint32_t k = 0; k < folds_; k++){
        left = right;
        if(k == folds_ - 1){  // the last right boundary should be the end of the set
            right = res_set.size();
        }
        else{
            right += res_set.size() / folds_;
        }

        sub_tra_set = res_set;
        sub_tra_set.erase(sub_tra_set.begin() + left, sub_tra_set.begin() + right); // leave out a portion of the training set for validation
        
        DecisionTreeClassifier dtc(sub_tra_set, n_classes, dtc_params_);
        for(uint32_t data_idx = left; data_idx < right; data_idx++){
            uint32_t label = res_set[data_idx][label_idx];
            std::vector<float> predict_prob = dtc.GetPredictProb(res_set[data_idx]);
            instance_hardnesses[data_idx] = (1.f - predict_prob[label]);
        }
    }

    std::vector<uint32_t> class_cnts(n_classes + 1, 0);
    for(uint32_t data_idx = 0; data_idx < res_set.size(); data_idx++){
        uint32_t label = res_set[data_idx][label_idx];
        class_cnts[label]++;
    }

    std::vector<std::vector<std::pair<uint32_t, float>>> ih_by_class(n_classes + 1); // instance hardnesses (ih)
    for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++){
        ih_by_class[class_idx].reserve(class_cnts[class_idx]);
    } 

    for(uint32_t data_idx = 0; data_idx < res_set.size(); data_idx++){
        uint32_t label = res_set[data_idx][label_idx];
        ih_by_class[label].emplace_back(data_idx, instance_hardnesses[data_idx]);
    }

    float num_data_to_preserved = *std::min_element(class_cnts.begin() + 1, class_cnts.end());

    std::vector<bool> is_preserved(res_set.size(), false);
    for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++){
        std::partial_sort(ih_by_class[class_idx].begin(), ih_by_class[class_idx].begin() + num_data_to_preserved, ih_by_class[class_idx].end(),
                            [](const std::pair<uint32_t, float> &a, const std::pair<uint32_t, float> &b){return a.second < b.second;}); // sort in ac
    
        for(uint32_t data_idx = 0; data_idx < num_data_to_preserved; data_idx++){
            is_preserved[ih_by_class[class_idx][data_idx].first] = true; // preserve easiest learning instances 
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
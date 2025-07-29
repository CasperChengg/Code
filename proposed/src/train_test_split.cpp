#include "../inc/train_test_split.h"

void train_test_split(const std::vector<std::vector<float>>&dataset, const float split_ratio,  std::vector<std::vector<float>> &tra_set, std::vector<std::vector<float>> &tst_set, const uint32_t n_classes)
{
    const uint32_t label_idx = dataset[0].size() - 1;    
    std::vector<uint32_t> class_cnts(n_classes + 1, 0); 
    for(uint32_t data_idx = 0; data_idx < dataset.size(); data_idx++){   
        uint32_t label = dataset[data_idx][label_idx];
        class_cnts[label]++;
    }

    std::vector<std::vector<uint32_t>> data_idxes_by_class(n_classes + 1);
    for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++){
        data_idxes_by_class[class_idx].reserve(class_cnts[class_idx]);
    }
    
    for(uint32_t data_idx = 0; data_idx < dataset.size(); data_idx++){
        uint32_t label = dataset[data_idx][label_idx];
        data_idxes_by_class[label].emplace_back(data_idx);
    }

    std::vector<bool> is_tra(dataset.size(), true);
    std::vector<bool> is_tst(dataset.size(), false);
    for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++){
        uint64_t time_seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::random_device rd;
        uint64_t seed = time_seed ^ (rd() << 1);
        shuffle(data_idxes_by_class[class_idx].begin(), data_idxes_by_class[class_idx].end(), 
                    std::default_random_engine(seed));

        uint32_t n_tst_data = ceil(data_idxes_by_class[class_idx].size() * (1.f - split_ratio));

        for(uint32_t idx = 0; idx < n_tst_data; idx++){
            uint32_t data_idx = data_idxes_by_class[class_idx][idx];    
            is_tst[data_idx] = true;
            if(data_idxes_by_class[class_idx].size() > 1){
                is_tra[data_idx] = false;
            }
        }

    }

    tra_set.reserve(ceil(dataset.size() * split_ratio));
    tst_set.reserve(ceil(dataset.size() * (1.f - split_ratio)));
    for(int data_idx = dataset.size() - 1; data_idx >= 0; data_idx--){
        if(is_tra[data_idx]){
            tra_set.emplace_back(dataset[data_idx]);
        }

        if(is_tst[data_idx]){
            tst_set.emplace_back(dataset[data_idx]);
        }
    }
}

void k_fold_split(const std::vector<std::vector<float>>& dataset, const uint32_t n_classes, const uint32_t k, std::vector<std::vector<std::vector<float>>> &tra_set, std::vector<std::vector<std::vector<float>>> &tst_set)
{
    const uint32_t label_idx = dataset[0].size() - 1;    
    std::vector<uint32_t> class_cnts(n_classes + 1, 0); 
    for(uint32_t data_idx = 0; data_idx < dataset.size(); data_idx++){   
        uint32_t label = dataset[data_idx][label_idx];
        class_cnts[label]++;
    }

    std::vector<std::vector<uint32_t>> data_idxes_by_class(n_classes + 1);
    for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++){
        data_idxes_by_class[class_idx].reserve(class_cnts[class_idx]);
    }
    
    for(uint32_t data_idx = 0; data_idx < dataset.size(); data_idx++){
        uint32_t label = dataset[data_idx][label_idx];
        data_idxes_by_class[label].push_back(data_idx);
    }

    tra_set.resize(k);
    tst_set.resize(k);

    std::random_device rd;
    uint64_t time_seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    uint64_t seed = time_seed ^ (rd() << 1);

    for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++){
        shuffle(data_idxes_by_class[class_idx].begin(), data_idxes_by_class[class_idx].end(), 
                                                                std::default_random_engine(seed));

        for(uint32_t shuffle_data_idx = 0; shuffle_data_idx < data_idxes_by_class[class_idx].size(); shuffle_data_idx++){          
            uint32_t data_idx = data_idxes_by_class[class_idx][shuffle_data_idx]; 

            uint32_t current_fold = 0;
            if(data_idxes_by_class[class_idx].size() <= 1){
                current_fold = k; // push into all folds
            }
            else{
                current_fold = shuffle_data_idx % k;
            }
            
            for(uint32_t fold_idx = 0; fold_idx < k; fold_idx++){
                if(fold_idx != current_fold){
                    tra_set[fold_idx].push_back(dataset[data_idx]);
                }
            }
            tst_set[current_fold].push_back(dataset[data_idx]);
        }
        if(data_idxes_by_class[class_idx].size() < k){
            uint32_t shuffle_data_idx = 0;
            for(uint32_t fold_idx = data_idxes_by_class[class_idx].size(); fold_idx < k; fold_idx++){
                uint32_t data_idx = data_idxes_by_class[class_idx][shuffle_data_idx]; 
                tst_set[fold_idx].push_back(dataset[data_idx]);
                shuffle_data_idx = (shuffle_data_idx + 1) % data_idxes_by_class[class_idx].size();
            }
        }
    }
}

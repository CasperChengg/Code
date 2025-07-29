#include "../inc/train_test_split.h"

static std::vector<uint32_t> CalculateClassCounts(const std::vector<std::vector<float>> &dataset, const uint32_t n_classes)
{
    // Class labels start from 1, so n_classes requires (n_classes + 1) space for direct indexing
    std::vector<uint32_t> class_counts(n_classes + 1, 0); 

    const uint32_t label_idx = dataset[0].size() - 1;
    for(uint32_t data_idx = 0; data_idx < dataset.size(); data_idx++){   
        uint32_t label = dataset[data_idx][label_idx];
        class_counts[label]++;
    }

    return class_counts;
}

void TrainTestSplit(const std::vector<std::vector<float>>&dataset, const float split_ratio,  std::vector<std::vector<float>> &training_set, std::vector<std::vector<float>> &testing_set, const uint32_t n_classes)
{
    const uint32_t label_idx = dataset[0].size() - 1;
    std::vector<uint32_t> data_idxes_by_class[n_classes + 1];
    
    std::vector<uint32_t> class_counts = CalculateClassCounts(dataset, n_classes);
    for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++){
        data_idxes_by_class[class_idx].reserve(class_counts[class_idx]);
    }
    
    for(uint32_t data_idx = 0; data_idx < dataset.size(); data_idx++){
        uint32_t label = dataset[data_idx][label_idx];
        data_idxes_by_class[label].push_back(data_idx);
    }

    std::vector<bool> is_training(dataset.size(), true);
    std::vector<bool> is_testing(dataset.size(), false);
    for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++){
        uint64_t time_seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::random_device rd;
        uint64_t seed = time_seed ^ (rd() << 1);
        shuffle(data_idxes_by_class[class_idx].begin(), data_idxes_by_class[class_idx].end(), 
                                                                std::default_random_engine(seed));

        uint32_t n_testing_data = ceil(data_idxes_by_class[class_idx].size() * (1.f - split_ratio));

        for(uint32_t shuffle_data_idx = 0; shuffle_data_idx < n_testing_data; shuffle_data_idx++){
            uint32_t data_idx = data_idxes_by_class[class_idx][shuffle_data_idx];    
            is_testing[data_idx] = true;
            if(data_idxes_by_class[class_idx].size() > 1){
                is_training[data_idx] = false;
            }
        }

    }

    training_set.reserve(dataset.size() * split_ratio);
    testing_set.reserve(dataset.size() * (1.f - split_ratio));
    for(int data_idx = dataset.size() - 1; data_idx >= 0; data_idx--){
        if(is_training[data_idx]){
            training_set.push_back(dataset[data_idx]);
        }

        if(is_testing[data_idx]){
            testing_set.push_back(dataset[data_idx]);
        }
    }
}

void KFoldSplit(const std::vector<std::vector<float>>& dataset, const uint32_t n_classes, const uint32_t k, std::vector<std::vector<std::vector<float>>> &training_set, std::vector<std::vector<std::vector<float>>> &testing_set)
{
    const uint32_t label_idx = dataset[0].size() - 1;
    std::vector<uint32_t> data_idxes_by_class[n_classes + 1];
    
    std::vector<uint32_t> class_counts = CalculateClassCounts(dataset, n_classes);
    for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++){
        data_idxes_by_class[class_idx].reserve(class_counts[class_idx]);
    }
    
    for(uint32_t data_idx = 0; data_idx < dataset.size(); data_idx++){
        uint32_t label = dataset[data_idx][label_idx];
        data_idxes_by_class[label].push_back(data_idx);
    }

    training_set.resize(k);
    testing_set.resize(k);

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
                    training_set[fold_idx].push_back(dataset[data_idx]);
                }
            }
            testing_set[current_fold].push_back(dataset[data_idx]);
        }
        if(data_idxes_by_class[class_idx].size() < k){
            uint32_t shuffle_data_idx = 0;
            for(uint32_t fold_idx = data_idxes_by_class[class_idx].size(); fold_idx < k; fold_idx++){
                uint32_t data_idx = data_idxes_by_class[class_idx][shuffle_data_idx]; 
                testing_set[fold_idx].push_back(dataset[data_idx]);
                shuffle_data_idx = (shuffle_data_idx + 1) % data_idxes_by_class[class_idx].size();
            }
        }
    }
}

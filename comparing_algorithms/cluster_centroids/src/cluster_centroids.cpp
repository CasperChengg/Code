#include "../inc/cluster_centroids.h"

std::vector<std::vector<float>> ClusterCentroids::fit_resample(std::vector<std::vector<float>> &tra_set, const uint32_t n_classes)
{
    const uint32_t label_idx = tra_set[0].size() - 1;
    std::vector<std::vector<float>> res_set = tra_set; // resample set

    std::vector<uint32_t> class_cnts(n_classes + 1, 0);
    for(uint32_t data_idx = 0; data_idx < res_set.size(); data_idx++){   
        uint32_t label = res_set[data_idx][label_idx];
        class_cnts[label]++;
    }

    std::vector<std::vector<float>> data_by_class[n_classes + 1];
    for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++){
        data_by_class[class_idx].reserve(class_cnts[class_idx]);
    }

    for(uint32_t data_idx = 0; data_idx < res_set.size(); data_idx++){
        uint32_t label = res_set[data_idx][label_idx];
        data_by_class[label].emplace_back(res_set[data_idx].begin(), res_set[data_idx].end() - 1); // exclude label
    }
    res_set.clear();

    const uint32_t num_data_to_preserve = *std::min_element(class_cnts.begin() + 1, class_cnts.end());
    for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++){
        KMeansPP kmpp(max_iters_, tolerance_);
        kmpp.fit(data_by_class[class_idx], num_data_to_preserve);
        std::vector<std::vector<float>> centroids = kmpp.get_centroids();
        
        // KMeans centroids have no label
        for(uint32_t centroid_idx = 0; centroid_idx < centroids.size(); centroid_idx++){
            centroids[centroid_idx].emplace_back(class_idx);
        }
        res_set.insert(res_set.end(), centroids.begin(), centroids.end());
    }
    
    class_cnts.assign(n_classes + 1, 0); // reset class counts
    for(uint32_t data_idx = 0; data_idx < res_set.size(); data_idx++){   
        uint32_t label = res_set[data_idx][label_idx];
        class_cnts[label]++;
    }


    return res_set;
}
#include "../inc/decision_tree_classifier.h"

#define IDX 0
#define VALUE 1
#define LABEL 2

float DecisionTreeClassifier::CustomRound(float x){
    return std::round(x * 1e6) / 1e6;
}

float DecisionTreeClassifier::CalculateGini(const std::vector<uint32_t> &left_partition_class_counts, const std::vector<uint32_t> &right_partition_class_counts)
{
    uint32_t left_partition_size = std::accumulate(left_partition_class_counts.begin() + 1, left_partition_class_counts.end(), 0);
    float left_partition_gini = 1.0;
    if(left_partition_size > 0){
        for(uint32_t class_idx = 1; class_idx < left_partition_class_counts.size(); class_idx++){
            float class_proportion = static_cast<float>(left_partition_class_counts[class_idx]) / left_partition_size;
            left_partition_gini -= class_proportion * class_proportion; 
        }
    }
   
    uint32_t right_partition_size = std::accumulate(right_partition_class_counts.begin() + 1, right_partition_class_counts.end(), 0);
    float right_partition_gini = 1.0;
    if(right_partition_size > 0){
        for(uint32_t class_idx = 1; class_idx < right_partition_class_counts.size(); class_idx++){
            float class_proportion = static_cast<float>(right_partition_class_counts[class_idx]) / right_partition_size;
            right_partition_gini -= class_proportion * class_proportion; 
        }
    }

    return static_cast<float>(left_partition_size) / (left_partition_size + right_partition_size) * left_partition_gini +
                static_cast<float>(right_partition_size) / (left_partition_size + right_partition_size) * right_partition_gini;
}

DecisionTreeClassifier::SplitPoint DecisionTreeClassifier::FindFeatureBestSplitPoint(const std::vector<std::vector<float>> &sorted_feature, const std::vector<bool> &is_existing_data)
{
    SplitPoint best_split_point = {0, 0.f, 1.1}; // Arbitrary feature field
    uint32_t best_left_idx = 0, best_right_idx = 0; 
    uint32_t left_idx = 0, right_idx = 0; // Split point = (left_value + right_value) / 2
    
    std::vector<uint32_t> left_partition_class_counts(n_classes + 1, 0);
    std::vector<uint32_t> right_partition_class_counts(n_classes + 1, 0);

    bool is_first_exist_data = true;
    for(uint32_t sorted_data_idx = 0; sorted_data_idx < sorted_feature.size(); sorted_data_idx++){
        uint32_t data_idx = sorted_feature[sorted_data_idx][IDX];
        if(is_existing_data[data_idx]){
            uint32_t data_label = sorted_feature[sorted_data_idx][LABEL];
            if(is_first_exist_data){ // move only first exist data which is the data with the smallest value in specific feature into left partition
                is_first_exist_data = false;
                left_partition_class_counts[data_label]++;
                right_idx = sorted_data_idx;
            }
            else{
                right_partition_class_counts[data_label]++;
            }
        }
    }

    float best_weighted_gini = 1.1;
    while(left_idx < sorted_feature.size()){
        left_idx = right_idx;
        for(right_idx = left_idx + 1; right_idx < sorted_feature.size(); right_idx++){
            uint32_t data_idx   = sorted_feature[right_idx][IDX];
            uint32_t data_label = sorted_feature[right_idx][LABEL];

            if(is_existing_data[data_idx]){
                float left_value  = sorted_feature[left_idx][VALUE];
                float right_value = sorted_feature[right_idx][VALUE];
                bool is_diff = CustomRound(left_value) != CustomRound(right_value); // CustomRound to address floating-point precision errors
                
                if(is_diff){
                    float weighted_gini = CalculateGini(left_partition_class_counts, right_partition_class_counts);
                    if(weighted_gini < best_weighted_gini){
                        best_weighted_gini = weighted_gini;
                        best_left_idx = left_idx;
                        best_right_idx = right_idx;
                    }
                    left_partition_class_counts[data_label]++;
                    right_partition_class_counts[data_label]--; 
                    break; 
                }
                left_partition_class_counts[data_label]++;
                right_partition_class_counts[data_label]--;    
            } 
        }
    }
    
    best_split_point.confidence = best_weighted_gini;
    best_split_point.value = (sorted_feature[best_left_idx][VALUE] + sorted_feature[best_right_idx][VALUE]) / 2;
   
    return best_split_point;
}

void DecisionTreeClassifier::FindBestSplitPoint(std::shared_ptr<TreeNode> node, const std::vector<std::vector<std::vector<float>>> &sorted_features, std::vector<bool> &is_existing_data)
{       
    std::vector<uint32_t> partition_class_counts((n_classes + 1), 0);
    for(uint32_t sorted_data_idx = 0; sorted_data_idx < sorted_features[0].size(); sorted_data_idx++){
        uint32_t data_idx = sorted_features[0][sorted_data_idx][IDX]; // Scaning one feature is enough
        if(is_existing_data[data_idx]){
            uint32_t data_label = sorted_features[0][sorted_data_idx][LABEL];
            partition_class_counts[data_label]++;
        }
    }
    
    auto max_it = std::max_element(partition_class_counts.begin() + 1, partition_class_counts.end());
    const uint32_t majority_label = std::distance(partition_class_counts.begin(), max_it);
    const uint32_t majority_count = *max_it;

    const uint32_t partition_size = std::accumulate(partition_class_counts.begin() + 1, partition_class_counts.end(), 0.f);
    const float purity = static_cast<float>(majority_count) / partition_size;
    if(partition_size <= dtc_param.min_samples_split || purity >= dtc_param.max_purity){ // stopping condition
        node->predict_prob.resize(n_classes + 1, 0.f);
        for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++){
            node->predict_prob[class_idx] = static_cast<float>(partition_class_counts[class_idx]) / partition_size;
        }
        return;
    } 

    node->split_point = {0, 0.f, 1.1};
    const uint32_t n_features = (sorted_features.size());
    for(uint32_t feature_idx = 0; feature_idx < n_features; feature_idx++){
        SplitPoint feature_best_split_point = FindFeatureBestSplitPoint(sorted_features[feature_idx], is_existing_data);
        if(node->split_point.confidence >= feature_best_split_point.confidence){
            node->split_point.feature = feature_idx;
            node->split_point.value = feature_best_split_point.value;
            node->split_point.confidence = feature_best_split_point.confidence;
        }
    }
    
    bool split_left_partition = false, split_right_partition = false;
    std::vector<bool> is_existing_data_in_left_partition(is_existing_data.size(), false);
    std::vector<bool> is_existing_data_in_right_partition(is_existing_data.size(), false);

    for(uint32_t sorted_data_idx = 0; sorted_data_idx < sorted_features[node->split_point.feature].size(); sorted_data_idx++){
        uint32_t data_idx = sorted_features[node->split_point.feature][sorted_data_idx][IDX];
        
        if(is_existing_data[data_idx]){
            float data_value = sorted_features[node->split_point.feature][sorted_data_idx][VALUE];
            if(data_value <= node->split_point.value){
                is_existing_data_in_left_partition[data_idx] = true;
                split_left_partition = true;
            }
            else{
                is_existing_data_in_right_partition[data_idx] = true;
                split_right_partition = true;
            }
        }
    }
    
    if(split_left_partition && split_right_partition){ // Split further only when both left and right partitions contain elements
        try{
            node->left_child = std::make_shared<TreeNode>(n_classes);
            node->right_child = std::make_shared<TreeNode>(n_classes); 
        }
        catch(const std::bad_alloc &error){
            printf("./%s:%d: error: %s\n", __FILE__, __LINE__, error.what());;
            exit(1);
        }

        FindBestSplitPoint(node->left_child, sorted_features, is_existing_data_in_left_partition);
        FindBestSplitPoint(node->right_child, sorted_features, is_existing_data_in_right_partition);        
    }
    else{
        node->predict_prob.resize(n_classes + 1, 0.f);
        for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++){
            node->predict_prob[class_idx] = static_cast<float>(partition_class_counts[class_idx]) / partition_size;
        }
    }
}

void DecisionTreeClassifier::CreateDecisionTree(const std::vector<std::vector<float>> &training_set)
{
    // dimension * data_size * 3(idx, value, label)
    const uint32_t n_dimensions = training_set[0].size() - 1; // except label
    std::vector<std::vector<std::vector<float>>> sorted_features(n_dimensions, std::vector<std::vector<float>>(training_set.size(), std::vector<float>(3, 0.f)));
    
    const uint32_t training_label_idx = training_set[0].size() - 1;
    for(uint32_t feature_idx = 0; feature_idx < training_set[0].size() - 1; feature_idx++){
        for(uint32_t training_data_idx = 0; training_data_idx < training_set.size(); training_data_idx++){
            sorted_features[feature_idx][training_data_idx][IDX]   = training_data_idx;
            sorted_features[feature_idx][training_data_idx][VALUE] = training_set[training_data_idx][feature_idx];
            sorted_features[feature_idx][training_data_idx][LABEL] = training_set[training_data_idx][training_label_idx];
        }
        std::sort(sorted_features[feature_idx].begin(), sorted_features[feature_idx].end(), 
                    [](const std::vector<float> &a, const std::vector<float> &b){return a[VALUE] < b[VALUE];});
    }

    std::vector<bool> is_existing_data(training_set.size(), true);
    FindBestSplitPoint(root, sorted_features, is_existing_data);
}

DecisionTreeClassifier::DecisionTreeClassifier(const std::vector<std::vector<float>> &training_set, const uint32_t n_classes, const struct decision_tree_parameter dtc_param)
                            :n_classes(n_classes), dtc_param(dtc_param)
{
    if(training_set.size() > 0){
        try{
            root = std::make_shared<TreeNode>(n_classes);
        }
        catch(const std::bad_alloc& error){
            printf("./%s:%d: error: %s\n", __FILE__, __LINE__, error.what());
            exit(1);
        }
        
        CreateDecisionTree(training_set);
    }
    else{
        printf("./%s:%d: error: empty training set\n", __FILE__, __LINE__);
        exit(1);
    }
}

std::vector<float> DecisionTreeClassifier::GetPredictProb(const std::vector<float> &testing_sample)
{
    std::shared_ptr<TreeNode> current_node = root;
    while(current_node->left_child != NULL && current_node->right_child != NULL){
        if(testing_sample[current_node->split_point.feature] <= current_node->split_point.value){
            current_node = current_node->left_child;
        }
        else{
            current_node = current_node->right_child;
        }
    }
    return current_node->predict_prob;
}

uint32_t DecisionTreeClassifier::GetPredictLabel(const std::vector<float> &testing_sample)
{
    std::vector<float> predict_prob = GetPredictProb(testing_sample);
    return std::distance(predict_prob.begin(), std::max_element(predict_prob.begin() + 1, predict_prob.end()));
}







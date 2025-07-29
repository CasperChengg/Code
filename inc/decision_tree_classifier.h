#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include <cmath> // std::round
#include <vector> // std::vector
#include <memory> // std::shared_ptr
#include <numeric> // std::accumulate
#include <algorithm> // std::max_element, std::sort
#include <iostream>

struct decision_tree_parameter{
    float max_purity;
    uint32_t min_samples_split;
};

class DecisionTreeClassifier{
    public:
        // Use in the training phase
        DecisionTreeClassifier(const std::vector<std::vector<float>> &training_set, 
                                    const uint32_t n_classes, 
                                        const struct decision_tree_parameter dtc_param);
        ~DecisionTreeClassifier(){root.reset();};

        // Use in the testing phase
        uint32_t GetPredictLabel(const std::vector<float> &testing_sample);
        std::vector<float> GetPredictProb(const std::vector<float> &testing_sample);
    
    private:
        class SplitPoint{
            public:
                uint32_t feature;
                float value;
                float confidence;
        }; // SplitPoint = {data[feature] <= value, confidence}

        class TreeNode{
            public:
                TreeNode(const uint32_t n_classes)
                {
                    split_point = {0, 0.f, 0.f};
                    predict_prob.resize(n_classes + 1, 0.f);
                    right_child = NULL;
                    left_child = NULL;
                };

                SplitPoint split_point;
                std::vector<float> predict_prob;
                std::shared_ptr<TreeNode> right_child;
                std::shared_ptr<TreeNode> left_child;
        };

        const uint32_t n_classes;
        const struct decision_tree_parameter dtc_param;
        
        std::shared_ptr<TreeNode> root;

        void CreateDecisionTree(const std::vector<std::vector<float>> &training_set);
        void FindBestSplitPoint(std::shared_ptr<TreeNode> node, const std::vector<std::vector<std::vector<float>>> &sorted_features, std::vector<bool> &is_existing_data);
        SplitPoint FindFeatureBestSplitPoint(const std::vector<std::vector<float>> &sorted_feature, const std::vector<bool> &is_existing_data);
        float CalculateGini(const std::vector<uint32_t> &class_counts_y, const std::vector<uint32_t> &class_counts_n);
        float CustomRound(float x);
};

#endif // DECISION_TREE_H
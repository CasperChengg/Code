#ifndef VALIDATION_H
#define VALIDATION_H

#include <cmath>  // pow
#include <string>
#include <vector> // std::vector
#include <limits> // std::numeric_limits
#include "../inc/decision_tree_classifier.h" // CreateDecisionTree, PredictByDecisionTree

#include <iostream>

class Validation{
    public:
        Validation(const std::vector<std::vector<float>> &training_set, const std::vector<std::vector<float>> &testing_set, const uint32_t n_classes, const decision_tree_parameter dtc_params, const bool macro_flag);
        ~Validation();

        float macro_precision;
        float macro_recall;
        float macro_f1;
        float g_mean;
        float MACC;
        float MAUC;
        float MMCC;
        float Cohens_Kappa;
        std::vector<std::vector<uint32_t>> confusion_matrix; 
    
    private:
        const uint32_t n_classes;

        std::vector<uint32_t> CalculateClassCounts(const std::vector<std::vector<float>> &training_set);
        void ConstructConfusionMatrix(const std::vector<std::vector<float>> &testing_set, DecisionTreeClassifier &dtc, const bool macro_flag = false);
        float CalculateOVRAUC (const std::vector<uint32_t> &ground_truth, const std::vector<std::vector<float>> &predict_prob, 
                                    const uint32_t pos_label);
        void ComputeMetrics(void);

};

#endif

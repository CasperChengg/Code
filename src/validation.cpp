#include "../inc/validation.h"

std::vector<uint32_t> Validation::CalculateClassCounts(const std::vector<std::vector<float>> &testing_set)
{
    std::vector<uint32_t> class_counts((n_classes + 1), 0); 

    const uint32_t label_idx = testing_set[0].size() - 1;
    for(uint32_t testing_data_idx = 0; testing_data_idx < testing_set.size(); testing_data_idx++){   
        uint32_t testing_data_label = testing_set[testing_data_idx][label_idx];
        class_counts[testing_data_label]++;
    }

    return class_counts;
}

float Validation::CalculateOVRAUC (const std::vector<uint32_t> &ground_truth, 
                                        const std::vector<std::vector<float>> &predict_prob, 
                                            const uint32_t pos_label) 
{
    std::vector<uint32_t> class_counts(n_classes + 1, 0);
    
    std::vector<std::pair<uint32_t, float>> data_label_with_pos_label_prob(ground_truth.size());
    for(uint32_t data_idx = 0; data_idx < ground_truth.size(); data_idx++){
       data_label_with_pos_label_prob[data_idx] = {ground_truth[data_idx], predict_prob[data_idx][pos_label]};
    }

    std::sort(data_label_with_pos_label_prob.begin(), data_label_with_pos_label_prob.end(), 
                [](const std::pair<uint32_t, float> &a, const std::pair<uint32_t, float> &b){return a.second > b.second;});
    
    float AUC = 0.f;
    float tp = 0.f, tp_prev = 0.f, fp = 0.f, fp_prev = 0.f;
    // Iteratively classify data as positive and compute (FP, TP) pairs to generate ROC points.
    for(uint32_t data_idx = 0; data_idx < ground_truth.size(); data_idx++){
        const uint32_t data_label = data_label_with_pos_label_prob[data_idx].first;
        class_counts[data_label]++;
        tp_prev = tp;
        fp_prev = fp;
        if(data_label == pos_label){
            tp++;
        }
        else{
            fp++;
        }

        // Integrate the ROC curve using trapezoid method
        AUC += (float)(tp + tp_prev) * (fp - fp_prev) / 2;
    }

    const uint32_t n_pos_label = class_counts[pos_label];
    const uint32_t n_neg_label = std::accumulate(class_counts.begin() + 1, class_counts.end(), 0) - n_pos_label;

    if(n_pos_label == 0 || n_neg_label == 0){
        return 0.f;
    }
    else{
        return AUC / (n_pos_label * n_neg_label);
    }
}

void Validation::ConstructConfusionMatrix(const std::vector<std::vector<float>> &testing_set, DecisionTreeClassifier &dtc, const bool macro_flag)
{
    const uint32_t label_idx = testing_set[0].size() - 1;
    std::vector<uint32_t> ground_truth(testing_set.size(), 0);
    std::vector<std::vector<float>> predict_prob(testing_set.size());

    // std::cerr << macro_flag << std::endl;
    for(uint32_t testing_data_idx = 0; testing_data_idx < testing_set.size(); testing_data_idx++){
        uint32_t testing_data_label = testing_set[testing_data_idx][label_idx]; // Ground truth
        ground_truth[testing_data_idx] = testing_data_label;

        predict_prob[testing_data_idx] = dtc.GetPredictProb(testing_set[testing_data_idx]);  
        uint32_t predicted_label = dtc.GetPredictLabel(testing_set[testing_data_idx]);  // Prediction
        confusion_matrix[predicted_label][testing_data_label]++;
        // std::cerr << predicted_label << ",";
    }
    // std::cerr << std::endl;

    // Compute OVRAUC here to avoid repeatedly passing ground_truth and predict_prob.
    uint32_t n_testing_classes = 0, minority_class_idx = 1;
    std::vector<uint32_t> class_counts = CalculateClassCounts(testing_set);
    for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++){
        if(class_counts[class_idx] > 0){ // n_testing_classes <= n_training_classes
            n_testing_classes++;
            float AUC = CalculateOVRAUC(ground_truth, predict_prob, class_idx);
            if(n_classes == 2 && !macro_flag){
                if(class_counts[class_idx] <= class_counts[minority_class_idx]){
                    MAUC = AUC;
                }
            }
            else{
                MAUC += AUC;
            }
        }
    }
    if(n_classes > 2 || macro_flag){
        MAUC /= n_testing_classes;
    }
}

void Validation::ComputeMetrics(void)
{ 
    uint32_t n_testing_data = 0;
    uint32_t n_testing_classes = 0; // n_testing_classes <= n_training_classes

    uint32_t n0 = 0, nc = 0; // Cohen's Kappa = (n0 / n - nc / n^2) / (1 - nc / n)

    uint32_t min_class_size = std::numeric_limits<uint32_t>::max();
    for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++){ 
        int64_t tp = 0, fp = 0, fn = 0, tn = 0; // uint32_t is not large enough to calculate MCC.
        uint32_t actual_class_count = 0, predict_class_count = 0;

        tp  = confusion_matrix[class_idx][class_idx];
        n0 += confusion_matrix[class_idx][class_idx];

        for(uint32_t col_idx = 1; col_idx <= n_classes; col_idx++){
            predict_class_count += confusion_matrix[class_idx][col_idx];
            tn += confusion_matrix[col_idx][col_idx];
        }
        fp  = predict_class_count - tp;
        tn -= tp;

        for(uint32_t row_idx = 1; row_idx <= n_classes; row_idx++){
            actual_class_count += confusion_matrix[row_idx][class_idx];
        }
        n_testing_data += actual_class_count;
        fn  = actual_class_count - tp;
        nc += predict_class_count * actual_class_count;

        if((tp + fn) > 0){ // Compute only if the class exists in the testing set.
            n_testing_classes++;

            float precision = 0.f;
            if((tp + fp) > 0){
                precision = (float)tp / (tp + fp);
            }

            float recall = 0.f;    
            if((tp + fn) > 0){
                recall = (float)tp / (tp + fn);
            }

            float f1_score = 0.f;
            if((precision + recall) > 0){
                f1_score = 2 * precision * recall / (precision + recall);
            }
            
            float ACC = (float)(tp + tn) / (tp + fp + fn + tn);

            float MCC = 0.f;
            if((tp + fp) > 0 && (tp + fn) > 0 && (tn + fp) > 0 && (tn + fn) > 0){
                MCC = (tp * tn - fp * fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
            }

            if(n_classes == 2){
                if(actual_class_count < min_class_size){
                    macro_precision = precision;
                    macro_recall    = recall;
                    macro_f1        = f1_score;
                    MACC            = ACC;
                    MMCC            = MCC;
                }
            }
            else{
                macro_precision += precision;
                macro_recall    += recall;
                macro_f1        += f1_score;
                MACC            += ACC;
                MMCC            += MCC;
            }
            g_mean *= recall;
        }
    }

    if(n_classes > 2){
        macro_precision /= n_testing_classes;
        macro_recall    /= n_testing_classes;
        macro_f1        /= n_testing_classes;
        MACC            /= n_testing_classes;
        MMCC            /= n_testing_classes;
    }
    g_mean = pow(g_mean, 1.f / n_testing_classes);

    float p0 = (float)n0 / n_testing_data;
    float pc = (float)nc / (n_testing_data * n_testing_data);
    Cohens_Kappa  = (p0 - pc) / (1 - pc);
}

Validation::Validation(const std::vector<std::vector<float>> &training_set, 
                        const std::vector<std::vector<float>> &testing_set, 
                            const uint32_t n_classes, 
                                const decision_tree_parameter dtc_params,
                                    const bool macro_flag)
            :n_classes(n_classes)
{       
    macro_precision = 0.f;
    macro_recall    = 0.f;
    macro_f1        = 0.f;
    g_mean          = 1.f;
    MACC            = 0.f;
    MAUC            = 0.f;
    MMCC            = 0.f;
    Cohens_Kappa    = 0.f;
    confusion_matrix.resize(n_classes + 1, std::vector<uint32_t>(n_classes + 1, 0));

    DecisionTreeClassifier dtc(training_set, n_classes, dtc_params);
    ConstructConfusionMatrix(testing_set, dtc, macro_flag);
    ComputeMetrics();
}

Validation::~Validation()
{
    std::vector<std::vector<uint32_t>>().swap(confusion_matrix);
}



#include <ctime> // timespec, clock_gettime
#include <iomanip> // std::fixed, std::setprecision
#include "../../../inc/validation.h" // Validation
#include "../../../inc/file_operations.h" // ReadTrainingAndTestingSet
#include "../inc/cluster_centroids.h"

int main(int argc, char *argv[])
{
   std::string file_path = "../../../datasets/" + (std::string)argv[1] + "-5-fold/" + (std::string)argv[1] + "-5-";
    std::string training_path = file_path + argv[2] + "tra.dat";
    std::string testing_path = file_path + argv[2] + "tst.dat";
    Dataset dataset = ReadTrainingAndTestingSet(training_path, testing_path);

    struct decision_tree_parameter dtc_params = {
        .max_purity = DTC_MAX_PURITY,
        .min_samples_split = DTC_MIN_SAMPLES_SPLIT
    };
    
    float running_time_ms = 0.f;
    timespec start_ns = {0}, end_ns = {0};
    clock_gettime(CLOCK_MONOTONIC, &start_ns);
    ClusterCentroids cc(CC_MAX_ITERS, CC_TOLERANCE);
    std::vector<std::vector<float>> resampled_set = cc.fit_resample(dataset.training_set, dataset.n_classes);
    Validation k_fold_validation(resampled_set, dataset.testing_set, dataset.n_classes, dtc_params, false);
    clock_gettime(CLOCK_MONOTONIC, &end_ns);
    running_time_ms = (float)(end_ns.tv_sec - start_ns.tv_sec) * 1000 + 
                                (float)(end_ns.tv_nsec - start_ns.tv_nsec) / 1000000;
    
    std::cout << std::fixed << std::setprecision(4) << k_fold_validation.macro_precision << std::endl;
    std::cout << std::fixed << std::setprecision(4) << k_fold_validation.macro_recall << std::endl;
    std::cout << std::fixed << std::setprecision(4) << k_fold_validation.macro_f1 << std::endl;
    std::cout << std::fixed << std::setprecision(4) << k_fold_validation.g_mean << std::endl;
    std::cout << std::fixed << std::setprecision(4) << k_fold_validation.MACC << std::endl;
    std::cout << std::fixed << std::setprecision(4) << k_fold_validation.MAUC << std::endl;
    std::cout << std::fixed << std::setprecision(4) << k_fold_validation.MMCC << std::endl;
    std::cout << std::fixed << std::setprecision(4) << k_fold_validation.Cohens_Kappa << std::endl;
    std::cout << std::fixed << std::setprecision(4) << running_time_ms << std::endl;
}
#!/bin/bash

declare -a synthetic_datasets=(
    "spiral"
    # "quantiles"
)

declare -a mul_real_world_datasets=(
    "vowel"
    # "segment"
    # "optdigits"
    # "penbased"
    # "vehicle"
    # "wine"
    # "hayes-roth"
    # "contraceptive"
    # "satimage"
    # "new-thyroid"
    # "dermatology"
    # "balance"
    # "glass"
    # "cleveland"
    # "thyroid"
    # "winequality-red"
    # "ecoli"
    # "yeast"
    # "pageblocks"
    # "winequality-white"
    # "shuttle"
)

declare -a bin_real_world_datasets=(
    # "vowel0"
    # "segment0"
    # "vehicle0"
    # "vehicle1"
    # "vehicle2"
    # "vehicle3"
    # "new-thyroid1"
    # "new-thyroid2"
    # "dermatology-6"
    # "glass0"
    # "glass1"
    # "glass2"
    # "glass4"
    # "glass5"
    # "glass6"
    # "winequality-red-4"
    # "ecoli1"
    # "ecoli2"
    # "ecoli3"
    # "ecoli4"
    # "yeast1"
    # "yeast3"
    # "yeast4"
)

# Output mode:
# 0 - Write results to a file in ./experiments/
# 1 - Print results to the console
DEBUG=0

# Number of runs for each dataset
NUM_RUNS=20
NUM_METRICS=9

# Parameters for Decision Tree Classifier
DTC_MIN_SAMPLES_SPLIT=10 
DTC_MAX_PURITY=0.95

if [ ! -d "./build" ]; then
    mkdir -p ./build
fi
cd build
CMAKE_OPTIONS="
    -DDTC_MIN_SAMPLES_SPLIT=${DTC_MIN_SAMPLES_SPLIT}
    -DDTC_MAX_PURITY=${DTC_MAX_PURITY}
"
cmake $CMAKE_OPTIONS ..
make

# ====================Run the synthetic datasets ====================
output_file_name=""
if [ "$DEBUG" -eq 0 ] && [ ${#synthetic_datasets[@]} -gt 0 ]; then
    if [ ! -d "../experiments" ]; then
        mkdir -p ../experiments
    fi
    output_file_name="../experiments/dtc_exp_syn.txt"
    >"$output_file_name"
    echo "Start: $(date +"%Y-%m-%d %H:%M:%S")" >> "$output_file_name"
    echo -e "NUM_RUNS = $NUM_RUNS\nMIN_SAMPLES_SPLIT = $DTC_MIN_SAMPLES_SPLIT\nMAX_PURITY        = $DTC_MAX_PURITY" >> "$output_file_name"
fi
for dataset in "${synthetic_datasets[@]}"; do
    metrics_sum=()
    for ((idx=0; idx<NUM_METRICS; idx++)); do
        metrics_sum[idx]=0
    done

    # Run the synthetic dataset NUM_RUNS times and accumulate each metric.
    for ((run=1; run<=$NUM_RUNS; run++)); do
        output=$(./main "$dataset" 1) # Synthetic datasets only have one fold.

        readarray -t metrics <<< "$output"
        for ((idx=0; idx<NUM_METRICS; idx++)); do
            metrics_sum[idx]=$(awk -v a="${metrics_sum[idx]}" -v b="${metrics[idx]}" 'BEGIN { printf "%.4f", a + b }')
        done
    done

    # Compute the average value of each metric.
    metrics_avg=()
    for ((idx=0; idx<NUM_METRICS; idx++)); do
        metrics_avg[idx]=$(awk -v a="${metrics_sum[idx]}" -v b="$NUM_RUNS" 'BEGIN { printf "%.4f", a / b }')
    done

    # Output the results
    if [ $DEBUG -eq 0 ]; then
        echo "========== $dataset ===========" >> "$output_file_name"
        for ((idx=0; idx<NUM_METRICS; idx++)); do
            echo "${metrics_avg[idx]}" >> "$output_file_name"
        done
    else
        echo "========== $dataset ==========="
        for ((idx=0; idx<NUM_METRICS; idx++)); do
            echo "${metrics_avg[idx]}"
        done
    fi
done

if [ "$DEBUG" -eq 0 ] && [ ${#synthetic_datasets[@]} -gt 0 ]; then
    echo "Finish: $(date +"%Y-%m-%d %H:%M:%S")" >> "$output_file_name"
fi

# ==================== Run the multi-class read-world datasets ====================
if [ "$DEBUG" -eq 0 ] && [ ${#mul_real_world_datasets[@]} -gt 0 ]; then
    output_file_name="../experiments/dtc_exp_mul.txt"
    >"$output_file_name"
    echo "Start: $(date +"%Y-%m-%d %H:%M:%S")" >> "$output_file_name"
    echo -e "NUM_RUNS = $NUM_RUNS\nMIN_SAMPLES_SPLIT = $DTC_MIN_SAMPLES_SPLIT\nMAX_PURITY        = $DTC_MAX_PURITY" >> "$output_file_name"
fi

for dataset in "${mul_real_world_datasets[@]}"; do
    metrics_sum=()
    for ((idx=0; idx<NUM_METRICS; idx++)); do
        metrics_sum[idx]=0
    done

    # Run the real-world dataset using k-fold cross-validation, repeating the process NUM_RUNS times, 
    # and accumulate each metric.
    for ((run=1; run<=$NUM_RUNS; run++)); do
        for((k=1; k<=5; k++)); do
            output=$(./main "$dataset" "$k") # Real-world datasets are partitioned into 5 folds.

            readarray -t metrics <<< "$output"
            for ((idx=0; idx<NUM_METRICS; idx++)); do
                metrics_sum[idx]=$(awk -v a="${metrics_sum[idx]}" -v b="${metrics[idx]}" 'BEGIN { printf "%.4f", a + b }')
            done
        done
    done

    # Compute the average value of each metric.
    metrics_avg=()
    for ((idx=0; idx<NUM_METRICS; idx++)); do
        metrics_avg[idx]=$(awk -v a="${metrics_sum[idx]}" -v b="$((5 * NUM_RUNS))" 'BEGIN { printf "%.4f", a / b }')
    done

    # Output the results
    if [ $DEBUG -eq 0 ]; then
        echo "========== $dataset ===========" >> "$output_file_name"
        for ((idx=0; idx<NUM_METRICS; idx++)); do
            echo "${metrics_avg[idx]}" >> "$output_file_name"
        done
    else
        echo "========== $dataset ==========="
        for ((idx=0; idx<NUM_METRICS; idx++)); do
            echo "${metrics_avg[idx]}"
        done
    fi
done

if [ "$DEBUG" -eq 0 ] && [ ${#mul_real_world_datasets[@]} -gt 0 ]; then
    echo "Finish: $(date +"%Y-%m-%d %H:%M:%S")" >> "$output_file_name"
fi

# ==================== Run the binary read-world datasets ====================
if [ "$DEBUG" -eq 0 ] && [ ${#bin_real_world_datasets[@]} -gt 0 ]; then
    output_file_name="../experiments/dtc_exp_bin.txt"
    >"$output_file_name"
    echo "Start: $(date +"%Y-%m-%d %H:%M:%S")" >> "$output_file_name"
    echo -e "NUM_RUNS = $NUM_RUNS\nMIN_SAMPLES_SPLIT = $DTC_MIN_SAMPLES_SPLIT\nMAX_PURITY        = $DTC_MAX_PURITY" >> "$output_file_name"
fi

for dataset in "${bin_real_world_datasets[@]}"; do
    metrics_sum=()
    for ((idx=0; idx<NUM_METRICS; idx++)); do
        metrics_sum[idx]=0
    done

    # Run the real-world dataset using k-fold cross-validation, repeating the process NUM_RUNS times, 
    # and accumulate each metric.
    for ((run=1; run<=$NUM_RUNS; run++)); do
        for((k=1; k<=5; k++)); do
            output=$(./main "$dataset" "$k") # Real-world datasets are partitioned into 5 folds.

            readarray -t metrics <<< "$output"
            for ((idx=0; idx<NUM_METRICS; idx++)); do
                metrics_sum[idx]=$(awk -v a="${metrics_sum[idx]}" -v b="${metrics[idx]}" 'BEGIN { printf "%.4f", a + b }')
            done
        done
    done

    # Compute the average value of each metric.
    metrics_avg=()
    for ((idx=0; idx<NUM_METRICS; idx++)); do
        metrics_avg[idx]=$(awk -v a="${metrics_sum[idx]}" -v b="$((5 * NUM_RUNS))" 'BEGIN { printf "%.4f", a / b }')
    done

    # Output the results
    if [ $DEBUG -eq 0 ]; then
        echo "========== $dataset ===========" >> "$output_file_name"
        for ((idx=0; idx<NUM_METRICS; idx++)); do
            echo "${metrics_avg[idx]}" >> "$output_file_name"
        done
    else
        echo "========== $dataset ==========="
        for ((idx=0; idx<NUM_METRICS; idx++)); do
            echo "${metrics_avg[idx]}"
        done
    fi
done

if [ "$DEBUG" -eq 0 ] && [ ${#bin_real_world_datasets[@]} -gt 0 ]; then
    echo "Finish: $(date +"%Y-%m-%d %H:%M:%S")" >> "$output_file_name"
fi

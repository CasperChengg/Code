#include "../inc/proposed.h"
inline static float euclidean_dist(const std::vector<float> &src, const std::vector<float> &dst)
{
    float squ_dist = 0;

    // the last column of vector stores the label
    for(uint32_t f_idx = 0; f_idx < src.size() - 1; f_idx++){
        float diff = src[f_idx] - dst[f_idx];
        squ_dist += diff * diff;
    }

    return sqrt(squ_dist);
}
void Proposed::rw_select_by_inf_scores(std::vector<bool> &selection_result, const uint32_t n_rounds)
{
    std::random_device rd;
    std::mt19937 gen(rd());

    float total_fitness = std::accumulate(inf_scores_.begin(), inf_scores_.end(), 0.f);
    for(uint32_t round = 0; round < n_rounds; round++){
        if(total_fitness <= 0.f){
            break;
        }

        std::uniform_real_distribution<> distrib(0.f, total_fitness);
        float random_float = distrib(gen);

        uint32_t selected_ind_idx;
        for(selected_ind_idx = 0; selected_ind_idx < res_set_->size(); selected_ind_idx++){
            random_float -= inf_scores_[selected_ind_idx];
            if(random_float <= 0){
                break;
            }
        }

        if(selected_ind_idx >= res_set_->size()){
            selected_ind_idx = res_set_->size() - 1; // Select the last individual in case of rounding errors
        }

        total_fitness -= inf_scores_[selected_ind_idx];
        inf_scores_[selected_ind_idx] = 0.f;
        selection_result[selected_ind_idx] = true;
    }
}

void Proposed::compute_inf_scores(const std::vector<std::vector<uint32_t>> &confusion_matrix)
{ 
    inf_scores_.resize(res_set_->size(), 0.f);

    std::vector<uint32_t> instance_queue;

    std::vector<bool> is_visited(res_set_->size());
    for(uint32_t src_idx = 0; src_idx < res_set_->size(); src_idx++){
        uint32_t src_label = (*res_set_)[src_idx][label_idx_];

        std::fill(is_visited.begin(), is_visited.end(), false);

        uint32_t n_pos_RNNs = 0, n_neg_RNNs = 0;
        float pos_dist_sum = 0.f, neg_dist_sum = 0.f;

        instance_queue.emplace_back(src_idx);
        is_visited[src_idx] = true;

        for(uint32_t level = 0; level < PROPOSED_LEVEL; level++){
            uint32_t current_q_size = instance_queue.size();
            for(uint32_t q_idx = 0; q_idx < current_q_size; q_idx++){
                uint32_t data_idx = instance_queue[q_idx];
                for(uint32_t idx = 0; idx < RNN[data_idx].size(); idx++){
                    uint32_t rnn_idx   = RNN[data_idx][idx];
                    uint32_t rnn_label = (*res_set_)[rnn_idx][label_idx_];
                    if(!is_visited[rnn_idx]){
                        if(level < (PROPOSED_LEVEL - 1)){ // last level is not expanded
                            instance_queue.emplace_back(rnn_idx);
                        }
                        is_visited[rnn_idx] = true;
 
                        if(rnn_label == src_label){
                            n_pos_RNNs++;
                            pos_dist_sum += (*dist_mat_)[src_idx][rnn_idx].second;
                        }
                        else{
                            float FNR_dst_to_src = static_cast<float>(confusion_matrix[src_label][rnn_label]) / 
                                                        class_cnts_[rnn_label];
                            float FNR_src_to_dst = static_cast<float>(confusion_matrix[rnn_label][src_label]) /
                                                        class_cnts_[src_label];
                            
                            if(FNR_dst_to_src > FNR_src_to_dst){
                                n_neg_RNNs++;
                                neg_dist_sum += (*dist_mat_)[src_idx][rnn_idx].second;
                            }
                        }
                    }
                }
            }
            instance_queue.erase(instance_queue.begin(), instance_queue.begin() + current_q_size);
        }
        instance_queue.clear();

        float pos_inf_score = 0.f, neg_inf_score = 0.f;
        if(n_pos_RNNs > 0 && pos_dist_sum > 0){
            float avg_pos_dist = pos_dist_sum / n_pos_RNNs;
            pos_inf_score = (float)(n_pos_RNNs) / (n_pos_RNNs + n_neg_RNNs) / avg_pos_dist;
        }
        if(n_neg_RNNs > 0 && neg_dist_sum > 0){
            float avg_neg_dist = neg_dist_sum / n_neg_RNNs;
            neg_inf_score = (float)(n_neg_RNNs) / (n_pos_RNNs + n_neg_RNNs) / avg_neg_dist;
        }
        
        float epsilon = 1e-7;
        inf_scores_[src_idx] = (neg_inf_score) / (pos_inf_score + epsilon) * log2(class_cnts_[src_label]);
    }
}

void Proposed::find_RNN(void)
{
    dist_mat_ = std::make_unique<std::vector<std::vector<std::pair<uint32_t, float>>>>(
        res_set_->size(),
        std::vector<std::pair<uint32_t, float>>(res_set_->size(), {0, 0.f})
    );

    for(uint32_t src_idx = 0; src_idx < res_set_->size(); src_idx++){
        (*dist_mat_)[src_idx][src_idx] = {src_idx, std::numeric_limits<float>::max()};
        for(uint32_t dst_idx = src_idx + 1; dst_idx < res_set_->size(); dst_idx++){
            float dist = euclidean_dist((*res_set_)[src_idx], (*res_set_)[dst_idx]);
            (*dist_mat_)[src_idx][dst_idx] = {dst_idx, dist};
            (*dist_mat_)[dst_idx][src_idx] = {src_idx, dist};
        }
    }
    
    RNN.resize(res_set_->size());
    for(uint32_t data_idx = 0; data_idx < res_set_->size(); data_idx++){
        uint32_t label = (*res_set_)[data_idx][label_idx_];
        RNN[data_idx].reserve(k_max_[label]);
    }

    for(uint32_t src_idx = 0; src_idx < res_set_->size(); src_idx++){
        const uint32_t src_label = (*res_set_)[src_idx][label_idx_];
        std::vector<std::pair<uint32_t, float>> knn(k_max_[src_label], {0, 0.f});
        std::partial_sort_copy((*dist_mat_)[src_idx].begin(), (*dist_mat_)[src_idx].end(),
                                    knn.begin(), knn.end(),          
                                        [](const std::pair<uint32_t, float> &a, const std::pair<uint32_t, float> &b){return a.second < b.second;});

        uint32_t n_same_class_nns = 0;
        for(uint32_t k = 0; k < k_max_[src_label]; k++){
            uint32_t nn_idx   = knn[k].first;
            uint32_t nn_label = (*res_set_)[nn_idx][label_idx_];

            if(nn_label == src_label){
                n_same_class_nns++;
            }

            // Adaptive k ranges from 3 to k_max for each sample.
            // Break if the ratio of same-class nearest neighbors to the current k is less than 0.5.
            if((float)n_same_class_nns / (k + 1) <= 0.5 && (k + 1) >= 3){ // scan nns >= 3
                break;
            } 
            
            RNN[nn_idx].emplace_back(src_idx);
        }
    }
}

void Proposed::compute_kmax(void)
{
    k_max_.resize(n_classes_ + 1, 0);
    for(uint32_t class_idx = 1; class_idx <= n_classes_; class_idx++){
        // Take square root of class count as adative k_max
        k_max_[class_idx] = ceil(sqrt(class_cnts_[class_idx]));
    }
}

std::vector<std::vector<float>> Proposed::fit_resample(const std::vector<std::vector<float>> &tra_set, const uint32_t n_classes)
{
    label_idx_ = tra_set[0].size() - 1;
    n_classes_ = n_classes;
    res_set_ = std::make_unique<std::vector<std::vector<float>>>(tra_set);

    class_cnts_.resize(n_classes + 1, 0);
    for(uint32_t data_idx = 0; data_idx < res_set_->size(); data_idx++){   
        uint32_t label = (*res_set_)[data_idx][label_idx_];
        class_cnts_[label]++;
    }

    if(*std::max_element(class_cnts_.begin() + 1, class_cnts_.end()) / *std::min_element(class_cnts_.begin() + 1, class_cnts_.end()) < 1.5){
        return *res_set_;
    }
     
    std::vector<std::vector<float>> pre_tra_set, pre_tst_set;
    train_test_split(tra_set, 0.7, pre_tra_set, pre_tst_set, n_classes_);
    Validation pre_valid(pre_tra_set, pre_tst_set, n_classes_, dtc_params_, true);

    compute_kmax();
    find_RNN();
    compute_inf_scores(pre_valid.confusion_matrix);

    uint32_t n_removed = res_set_->size() * (1 - pre_valid.MAUC);
    uint32_t n_removed_candi = std::count_if(inf_scores_.begin(), inf_scores_.end(), 
                                                    [](float score){return score > 0.f;}); 
    if(n_removed > n_removed_candi){
        n_removed = n_removed_candi;
    }
    std::vector<bool> is_removed(tra_set.size(), false);
    rw_select_by_inf_scores(is_removed, n_removed);
    
    for(int data_idx = (res_set_->size() - 1); data_idx >= 0; data_idx--){
        if(is_removed[data_idx]){
            res_set_->erase(res_set_->begin() + data_idx);
        }
    }

    return *res_set_;
}




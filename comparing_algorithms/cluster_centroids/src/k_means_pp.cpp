#include "../inc/k_means_pp.h"

static inline float euclidean_dist(std::vector<float> &src, std::vector<float> &dst)
{
    const uint32_t n_dim = src.size();
    float squ_dist = 0;

    for(uint32_t dim_idx = 0; dim_idx < n_dim; dim_idx++)
    {
        float diff = src[dim_idx] - dst[dim_idx];
        squ_dist += diff * diff;
    }

    return sqrt(squ_dist);
}

uint32_t KMeansPP::rw_selection(std::vector<float> fitnesses)
{
    float total_fitness = std::accumulate(fitnesses.begin(), fitnesses.end(), 0.f);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(0.0, total_fitness);

    float random_value = distrib(gen);
    for(uint32_t fitness_idx = 0; fitness_idx < fitnesses.size(); fitness_idx++){
        random_value -= fitnesses[fitness_idx];
        if(random_value <= 0.f){
            return fitness_idx;
        }
    }

    return fitnesses.size() - 1;
}

void KMeansPP::gen_init_centroids(uint32_t n_clusters)
{
    std::vector<std::vector<float>> dist_mat(res_set_->size(), std::vector<float>(res_set_->size(), 0.f));
    for(uint32_t src_idx = 0; src_idx < res_set_->size(); src_idx++){
        dist_mat[src_idx][src_idx] = 0;
        for(uint32_t dst_idx = (src_idx + 1); dst_idx < res_set_->size(); dst_idx++){
            float dist = euclidean_dist((*res_set_)[src_idx], (*res_set_)[dst_idx]);
            dist_mat[src_idx][dst_idx] = dist;
            dist_mat[dst_idx][src_idx] = dist;
        }
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, res_set_->size() - 1);
    
    std::vector<uint32_t> init_centroids_idx(n_clusters); 
    init_centroids_idx[0] = distrib(gen); // Random select first centroid
    
    std::vector<float> fitnesses(res_set_->size(), 0.f);
    for(uint32_t centroid_idx = 1; centroid_idx < n_clusters; centroid_idx++){
        fitnesses.assign(res_set_->size(), 0.f);
        for(uint32_t data_idx = 0; data_idx < res_set_->size();data_idx++){
            for(uint32_t known_centroid_idx = 0; known_centroid_idx < centroid_idx; known_centroid_idx++){
                uint32_t dst_idx = init_centroids_idx[known_centroid_idx];
                fitnesses[data_idx] += dist_mat[data_idx][dst_idx];   
            }
        }

        uint32_t selected_idx = rw_selection(fitnesses);
        init_centroids_idx[centroid_idx] = selected_idx;
    }

    centroids_.clear();
    centroids_.reserve(n_clusters);
    for(uint32_t centroid_idx = 0; centroid_idx < n_clusters; centroid_idx++){
        centroids_.emplace_back((*res_set_)[init_centroids_idx[centroid_idx]]);
    }
}

void KMeansPP::fit(const std::vector<std::vector<float>> &tra_set, const uint32_t n_clusters)
{
    res_set_ = std::make_unique<std::vector<std::vector<float>>>(tra_set);

    uint32_t n_iter = 0;

    std::vector<uint32_t> label(res_set_->size(), 0);
    std::vector<uint32_t> cluster_cnts(n_clusters, 0);

    float previous_SSE = 0.f, current_SSE = std::numeric_limits<float>::max();

    gen_init_centroids(n_clusters);

    uint32_t n_features = (*res_set_)[0].size();
    do{
        previous_SSE = current_SSE;
        current_SSE  = 0;

        cluster_cnts.assign(n_clusters, 0);
        for(uint32_t data_idx = 0; data_idx < res_set_->size(); data_idx++){
            uint32_t nearest_centroid_idx = 0;
            float distance_to_nearest_centroid = std::numeric_limits<float>::max();
            for(uint32_t cluster_idx = 0; cluster_idx < n_clusters; cluster_idx++){
                float distance_to_centroid = euclidean_dist((*res_set_)[data_idx], centroids_[cluster_idx]);
                if(distance_to_centroid < distance_to_nearest_centroid){
                    nearest_centroid_idx         = cluster_idx;
                    distance_to_nearest_centroid = distance_to_centroid;
                }
            }
            label[data_idx] = nearest_centroid_idx;
            cluster_cnts[label[data_idx]]++;
            current_SSE += distance_to_nearest_centroid * distance_to_nearest_centroid;
        }

        centroids_.assign(n_clusters, std::vector<float>(n_features, 0.f));
        for(uint32_t data_idx = 0; data_idx < res_set_->size(); data_idx++){  
            for(uint32_t feature_idx = 0; feature_idx < n_features; feature_idx++){
                centroids_[label[data_idx]][feature_idx] += (*res_set_)[data_idx][feature_idx];
            }
        }

        for(uint32_t centroid_idx = 0; centroid_idx < n_clusters; centroid_idx++){  
            for(uint32_t feature_idx = 0; feature_idx < n_features; feature_idx++){
                centroids_[centroid_idx][feature_idx] /= cluster_cnts[centroid_idx];
            }
        }

        if(++n_iter > max_iter_){
            break;
        }
    }
    while((previous_SSE - current_SSE) > tolerance_);

    for(int centroid_idx = n_clusters - 1; centroid_idx >= 0; centroid_idx--){
        if(std::isnan(centroids_[centroid_idx][0])){
            centroids_.erase(centroids_.begin() + centroid_idx);
        }
    }
}



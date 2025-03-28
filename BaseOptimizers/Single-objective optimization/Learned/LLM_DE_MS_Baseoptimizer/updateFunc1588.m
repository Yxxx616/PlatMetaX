% MATLAB Code
function [offspring] = updateFunc1588(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Cluster the population (K=3)
    K = 3;
    centroids = zeros(K, D);
    cluster_idx = zeros(NP, 1);
    dist_matrix = pdist2(popdecs, popdecs);
    
    % Initialize centroids with diverse solutions
    [~, max_idx] = max(dist_matrix(:));
    [r1, r2] = ind2sub([NP, NP], max_idx);
    centroids(1,:) = popdecs(r1,:);
    centroids(2,:) = popdecs(r2,:);
    centroids(3,:) = mean(popdecs, 1);
    
    % Simple clustering
    for iter = 1:5
        dist_to_centroids = pdist2(popdecs, centroids);
        [~, cluster_idx] = min(dist_to_centroids, [], 2);
        for k = 1:K
            members = popdecs(cluster_idx == k, :);
            if ~isempty(members)
                centroids(k,:) = mean(members, 1);
            end
        end
    end
    
    % 2. Identify elite and nearest centroids
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible,:);
        elite = elite(elite_idx,:);
    else
        [~, elite_idx] = min(popfits + 1e6*max(0, cons));
        elite = popdecs(elite_idx,:);
    end
    
    % Find nearest centroid for each solution
    dist_to_centroids = pdist2(popdecs, centroids);
    [~, nearest_centroid] = min(dist_to_centroids, [], 2);
    
    % 3. Compute direction vectors
    dir_vectors = zeros(NP, D);
    alpha = 0.5 + 0.3*rand(NP,1);
    beta = 0.3 + 0.2*rand(NP,1);
    gamma = 0.2 + 0.2*rand(NP,1);
    
    rand_idx1 = randi(NP, NP, 1);
    rand_idx2 = randi(NP, NP, 1);
    same_idx = rand_idx1 == rand_idx2;
    rand_idx2(same_idx) = mod(rand_idx2(same_idx) + randi(NP-1, sum(same_idx), NP) + 1;
    
    for i = 1:NP
        nearest_cent = centroids(nearest_centroid(i),:);
        dir_vectors(i,:) = alpha(i)*(elite - popdecs(i,:)) + ...
                          beta(i)*(nearest_cent - popdecs(i,:)) + ...
                          gamma(i)*(popdecs(rand_idx1(i),:) - popdecs(rand_idx2(i),:));
    end
    
    % 4. Adaptive scaling factors based on ranks
    [~, ranks] = sort(popfits);
    F = 0.4 + 0.3*(1 - ranks/NP);
    
    % 5. Personal best (top 30% solutions)
    [~, sorted_idx] = sort(popfits);
    pbest_pool = popdecs(sorted_idx(1:ceil(NP*0.3)), :);
    pbest = pbest_pool(randi(size(pbest_pool,1), NP, 1);
    
    % 6. Generate mutation vectors
    eta = 0.05 * rand(NP,1);
    offspring = popdecs + F.*dir_vectors + eta.*(pbest - popdecs);
    
    % 7. Rank-based crossover
    CR = 0.7 + 0.2*(1 - ranks/NP);
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    offspring = popdecs .* (~mask) + offspring .* mask;
    
    % 8. Boundary handling with reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = 2*lb(lb_mask) - offspring(lb_mask);
    offspring(ub_mask) = 2*ub(ub_mask) - offspring(ub_mask);
    
    % 9. Final boundary enforcement
    offspring = min(max(offspring, lb), ub);
end
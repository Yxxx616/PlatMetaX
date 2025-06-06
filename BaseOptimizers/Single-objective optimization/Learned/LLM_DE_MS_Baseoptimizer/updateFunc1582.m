% MATLAB Code
function [offspring] = updateFunc1582(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Compute weighted centroid
    f_mean = mean(popfits);
    f_std = max(std(popfits), 1e-6);
    weights = 1 ./ (1 + exp((popfits - f_mean)/f_std));
    weights = weights ./ sum(weights);
    centroid = weights' * popdecs;
    
    % 2. Identify best solution (considering constraints)
    feasible = cons <= 0;
    if any(feasible)
        [~, best_idx] = min(popfits(feasible));
        x_best = popdecs(best_idx, :);
    else
        [~, best_idx] = min(popfits + 1e6*max(0,cons));
        x_best = popdecs(best_idx, :);
    end
    
    % 3. Compute personal best (top 20% solutions)
    [~, sorted_idx] = sort(popfits);
    pbest_pool = popdecs(sorted_idx(1:ceil(NP*0.2)), :);
    pbest = pbest_pool(randi(size(pbest_pool,1), NP, 1), :);
    
    % 4. Compute adaptive direction vectors
    dir_vectors = zeros(NP, D);
    alpha = 0.6 + 0.4*rand(NP,1);
    beta = 0.3 * rand(NP,1);
    rand_vec = randn(NP,D);
    
    for i = 1:NP
        if cons(i) <= 0
            dir_vectors(i,:) = x_best - popdecs(i,:);
        else
            dir_vectors(i,:) = alpha(i)*(centroid - popdecs(i,:)) + beta(i)*rand_vec(i,:);
        end
    end
    
    % 5. Compute adaptive scaling factors
    c_norm = max(0, cons) ./ (max(abs(cons)) + 1e-6);
    f_min = min(popfits); f_max = max(popfits);
    F = 0.4 + 0.4*(1 - c_norm) .* (1 - (popfits - f_min)/(f_max - f_min + 1e-6));
    F = min(max(F, 0.1), 0.8);
    
    % 6. Generate mutation vectors
    eta = 0.05 * rand(NP,1);
    offspring = popdecs + F.*dir_vectors + eta.*(pbest - popdecs);
    
    % 7. Rank-based crossover
    [~, ranks] = sort(popfits);
    CR = 0.1 + 0.7 * (ranks/NP);
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    offspring = popdecs .* (~mask) + offspring .* mask;
    
    % 8. Improved boundary handling
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    r = rand(NP,D);
    offspring(lb_mask) = popdecs(lb_mask) + r(lb_mask) .* (lb(lb_mask) - popdecs(lb_mask));
    offspring(ub_mask) = popdecs(ub_mask) + r(ub_mask) .* (ub(ub_mask) - popdecs(ub_mask));
    
    % 9. Final boundary enforcement
    offspring = min(max(offspring, lb), ub);
end
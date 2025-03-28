% MATLAB Code
function [offspring] = updateFunc1725(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Process constraints
    pos_cons = max(0, cons);
    feasible_mask = pos_cons == 0;
    
    % Rank-based weights (exponential decay)
    [~, sorted_idx] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(sorted_idx) = 1:NP;
    rank_weights = exp(-5*ranks/NP);
    rank_weights = rank_weights / sum(rank_weights);
    
    % Select base vector (best feasible or least violated)
    if any(feasible_mask)
        feas_pool = find(feasible_mask);
        [~, best_feas_idx] = min(popfits(feasible_mask));
        base_vec = popdecs(feas_pool(best_feas_idx),:);
    else
        [~, least_viol_idx] = min(pos_cons);
        base_vec = popdecs(least_viol_idx,:);
    end
    
    % Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fits = (popfits - f_min) / (f_max - f_min + eps);
    
    c_max = max(pos_cons);
    norm_cons = pos_cons / (c_max + eps);
    
    % Compute population mean and weighted differences
    pop_mean = mean(popdecs, 1);
    weighted_diff = zeros(NP, D);
    for i = 1:NP
        weighted_diff(i,:) = sum(bsxfun(@times, popdecs - pop_mean, rank_weights), 1);
    end
    
    % Adaptive parameters
    F = 0.5 + 0.3 * tanh(10 * norm_fits);
    alpha = 0.2 * (1 - exp(-5 * norm_cons));
    CR = 0.9 - 0.5 * norm_cons;
    
    % Mutation with adaptive components
    mutation = bsxfun(@plus, base_vec, bsxfun(@times, F, weighted_diff)) + ...
               bsxfun(@times, alpha, randn(NP, D));
    
    % Binomial crossover with jitter
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) <= CR(:, ones(1,D)) | bsxfun(@eq, (1:D), j_rand);
    offspring = popdecs;
    offspring(mask) = mutation(mask);
    
    % Boundary handling with random reset for high violations
    lb_matrix = lb(ones(NP,1),:);
    ub_matrix = ub(ones(NP,1),:);
    
    high_viol = norm_cons > 0.8;
    if any(high_viol)
        rand_mask = bsxfun(@and, high_viol, rand(NP,D) < 0.3);
        offspring(rand_mask) = lb_matrix(rand_mask) + ...
                             rand(sum(rand_mask(:)),1) .* ...
                             (ub_matrix(rand_mask) - lb_matrix(rand_mask));
    end
    
    % Ensure boundaries
    offspring = min(max(offspring, lb_matrix), ub_matrix);
    
    % Elite preservation
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
        offspring(feas_pool(best_idx),:) = popdecs(feas_pool(best_idx),:);
    end
end
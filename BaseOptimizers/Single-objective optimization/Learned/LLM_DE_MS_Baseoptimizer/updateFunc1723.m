% MATLAB Code
function [offspring] = updateFunc1723(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Process constraints
    pos_cons = max(0, cons);
    feasible_mask = pos_cons == 0;
    
    % Identify best solutions
    [~, sorted_fit_idx] = sort(popfits);
    rank_weights = exp(-5*(1:NP)'/NP);
    rank_weights = rank_weights / sum(rank_weights);
    
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
    
    % Compute population mean
    pop_mean = mean(popdecs, 1);
    
    % Adaptive parameters
    F = 0.5 + 0.3 * tanh(10 * norm_fits);
    alpha = 0.2 * (1 - exp(-5 * norm_cons));
    CR = 0.9 - 0.5 * norm_cons;
    
    % Initialize offspring
    offspring = zeros(NP, D);
    j_rand = randi(D, NP, 1);
    
    % Vectorized mutation
    weighted_diff = zeros(NP, D);
    for i = 1:NP
        weighted_diff(i,:) = sum(bsxfun(@times, popdecs - pop_mean, rank_weights), 1);
    end
    
    mutation = bsxfun(@plus, base_vec, bsxfun(@times, F, weighted_diff)) + ...
               bsxfun(@times, alpha .* (1 + norm_cons), randn(NP, D));
    
    % Binomial crossover
    mask = rand(NP, D) <= CR(:, ones(1,D)) | bsxfun(@eq, (1:D), j_rand);
    offspring = popdecs;
    offspring(mask) = mutation(mask);
    
    % Boundary handling
    lb_matrix = lb(ones(NP,1),:);
    ub_matrix = ub(ones(NP,1),:);
    
    % For highly constrained solutions, use random reinitialization
    high_viol = norm_cons > 0.8;
    if any(high_viol)
        rand_mask = bsxfun(@and, high_viol, rand(NP,D) < 0.3);
        offspring(rand_mask) = lb_matrix(rand_mask) + ...
                             rand(sum(rand_mask(:)),1) .* ...
                             (ub_matrix(rand_mask) - lb_matrix(rand_mask));
    end
    
    % Standard boundary check
    offspring = min(max(offspring, lb_matrix), ub_matrix);
    
    % Additional local search for best solutions
    if any(feasible_mask)
        [~, top3] = mink(popfits(feasible_mask), 3);
        for i = 1:min(3, length(top3))
            idx = feas_pool(top3(i));
            offspring(idx,:) = offspring(idx,:) + 0.1*(ub-lb).*randn(1,D);
            offspring(idx,:) = min(max(offspring(idx,:), lb), ub);
        end
    end
end
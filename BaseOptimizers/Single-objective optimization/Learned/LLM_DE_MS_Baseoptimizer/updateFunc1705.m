% MATLAB Code
function [offspring] = updateFunc1705(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Process constraints (positive values indicate violation)
    pos_cons = max(0, cons);
    feasible_mask = pos_cons == 0;
    
    % Identify best solutions
    [~, best_idx] = min(popfits);
    if any(feasible_mask)
        feas_pool = find(feasible_mask);
        [~, best_feas_idx] = min(popfits(feasible_mask));
        best_feas_idx = feas_pool(best_feas_idx);
        base_vec = popdecs(best_feas_idx,:);
    else
        [~, best_feas_idx] = min(pos_cons);
        base_vec = popdecs(best_feas_idx,:);
    end
    
    % Normalize fitness and constraints for adaptation
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fits = (popfits - f_min) / (f_max - f_min + eps);
    
    c_min = min(pos_cons);
    c_max = max(pos_cons);
    norm_cons = (pos_cons - c_min) / (c_max - c_min + eps);
    
    % Enhanced adaptive parameters
    alpha = 0.8 - 0.3 * norm_cons;
    F = 0.6 + 0.2 * (1 - norm_fits) .* (1 - norm_cons);
    beta = 0.4 * (1 - norm_cons);
    CR = 0.95 - 0.25 * norm_fits;
    
    % Initialize offspring
    offspring = zeros(NP, D);
    j_rand = randi(D, NP, 1);
    
    % Pre-generate random indices (4 per individual)
    idx_matrix = zeros(NP, 4);
    for i = 1:NP
        idx_pool = setdiff(1:NP, i);
        idx_matrix(i,:) = idx_pool(randperm(length(idx_pool), 4));
    end
    
    % Weighted base vector
    weighted_base = alpha .* popdecs(best_idx,:) + (1-alpha) .* base_vec;
    
    % Vectorized mutation and crossover
    for i = 1:NP
        r = idx_matrix(i,:);
        mutation = weighted_base(i,:) + F(i) * (popdecs(r(1),:) - popdecs(r(2),:)) + ...
                  beta(i) * (popdecs(r(3),:) - popdecs(r(4),:));
        
        % Binomial crossover
        mask = rand(1,D) <= CR(i) | (1:D) == j_rand(i);
        offspring(i,:) = popdecs(i,:);
        offspring(i,mask) = mutation(mask);
        
        % Boundary handling with reflection
        viol_low = offspring(i,:) < lb;
        viol_high = offspring(i,:) > ub;
        offspring(i,viol_low) = 2*lb(viol_low) - offspring(i,viol_low);
        offspring(i,viol_high) = 2*ub(viol_high) - offspring(i,viol_high);
    end
    
    % Adaptive perturbation for diversity
    diversity_mask = (norm_fits + norm_cons) > 0.75;
    if any(diversity_mask)
        perturb_amount = 0.15 * (rand(sum(diversity_mask),D) .* (ub - lb);
        offspring(diversity_mask,:) = offspring(diversity_mask,:) + perturb_amount;
        offspring = min(max(offspring, lb), ub);
    end
end
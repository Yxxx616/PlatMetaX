% MATLAB Code
function [offspring] = updateFunc1695(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Process constraints (positive values indicate violation)
    pos_cons = max(0, cons);
    c_min = min(pos_cons);
    c_max = max(pos_cons);
    w_cons = (pos_cons - c_min) / (c_max - c_min + eps);
    
    % Normalize fitness (0 is best)
    f_min = min(popfits);
    f_max = max(popfits);
    w_fit = (popfits - f_min) / (f_max - f_min + eps);
    
    % Identify best solutions
    [~, best_idx] = min(popfits);
    feasible_mask = pos_cons == 0;
    if any(feasible_mask)
        feas_pool = find(feasible_mask);
        [~, best_feas_idx] = min(popfits(feasible_mask));
        best_feas_idx = feas_pool(best_feas_idx);
    else
        [~, best_feas_idx] = min(pos_cons);
    end
    
    % Initialize offspring
    offspring = zeros(NP, D);
    
    % Pre-generate random numbers for vectorization
    F = 0.4 + 0.3 * cos(pi * w_fit) .* (1 - w_cons);
    CR = 0.1 + 0.8 * (1 - w_fit) .* (1 - 0.5 * w_cons);
    j_rand = randi(D, NP, 1);
    
    for i = 1:NP
        % Select distinct random vectors (excluding current)
        idx_pool = setdiff(1:NP, i);
        r = idx_pool(randperm(length(idx_pool), 3));
        
        % Adaptive coefficients
        alpha = 0.5 * (1 + w_cons(i));
        beta = 0.3 * (1 - w_fit(i));
        
        % Constraint-aware mutation
        mutation = popdecs(r(1),:) + F(i) * (popdecs(r(2),:) - popdecs(r(3),:)) + ...
                  alpha * (popdecs(best_idx,:) - popdecs(i,:)) + ...
                  beta * (popdecs(best_feas_idx,:) - popdecs(i,:));
        
        % Binomial crossover
        mask = rand(1,D) <= CR(i) | (1:D) == j_rand(i);
        offspring(i,:) = popdecs(i,:);
        offspring(i,mask) = mutation(mask);
        
        % Boundary handling with adaptive reflection
        viol_low = offspring(i,:) < lb;
        viol_high = offspring(i,:) > ub;
        offspring(i,viol_low) = lb(viol_low) + 0.5 * (popdecs(i,viol_low) - lb(viol_low));
        offspring(i,viol_high) = ub(viol_high) - 0.5 * (ub(viol_high) - popdecs(i,viol_high));
    end
    
    % Final adaptive perturbation for highly constrained solutions
    perturb_mask = rand(NP,D) < repmat(0.1 * w_cons, 1, D);
    offspring(perturb_mask) = offspring(perturb_mask) + ...
                            0.1 * (ub(perturb_mask)-lb(perturb_mask)) .* randn(sum(perturb_mask(:)),1);
    offspring = min(max(offspring, lb), ub);
end
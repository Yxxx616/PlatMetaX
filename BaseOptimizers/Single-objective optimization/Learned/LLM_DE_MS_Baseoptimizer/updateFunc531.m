% MATLAB Code
function [offspring] = updateFunc531(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fit = (popfits - f_min) / (f_max - f_min + eps);
    
    abs_cons = abs(cons);
    c_min = min(abs_cons);
    c_max = max(abs_cons);
    norm_con = (abs_cons - c_min) / (c_max - c_min + eps);
    
    % Adaptive weights
    w = 0.7 * norm_fit + 0.3 * norm_con;
    
    % Select elite individual and feasible pool
    feasible_mask = cons <= 0;
    feasible_pop = popdecs(feasible_mask,:);
    feasible_fits = popfits(feasible_mask);
    
    if ~isempty(feasible_pop)
        [~, elite_idx] = min(feasible_fits);
        elite = feasible_pop(elite_idx,:);
    else
        [~, elite_idx] = min(popfits + 10*norm_con);
        elite = popdecs(elite_idx,:);
    end
    
    % Generate random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx);
    r2 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx);
    
    % Select feasible individuals if available
    if ~isempty(feasible_pop)
        feasible_idx = randi(size(feasible_pop,1), NP, 1);
        feasible_diff = feasible_pop(feasible_idx,:) - popdecs;
    else
        feasible_diff = popdecs(r1,:) - popdecs(r2,:);
    end
    
    % Calculate direction vectors
    elite_diff = repmat(elite, NP, 1) - popdecs;
    rand_diff = popdecs(r1,:) - popdecs(r2,:);
    
    % Adaptive scaling factors
    F1 = 0.8 * (1 - w);
    F2 = 0.6 * w .* (1 - norm_con);
    F3 = 0.4 * norm_con;
    
    % Mutation operation
    offspring = popdecs + repmat(F1, 1, D) .* elite_diff + ...
                repmat(F2, 1, D) .* feasible_diff + ...
                repmat(F3, 1, D) .* rand_diff;
    
    % Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Reflect back into bounds if out of range
    mask_low = offspring < lb_rep;
    mask_high = offspring > ub_rep;
    offspring = offspring .* ~(mask_low | mask_high) + ...
                (2*lb_rep - offspring) .* mask_low + ...
                (2*ub_rep - offspring) .* mask_high;
    
    % Small adaptive perturbation
    perturb = randn(NP,D) .* repmat(0.05*(1-w),1,D) .* (ub_rep-lb_rep);
    offspring = offspring + perturb;
    
    % Final boundary check
    offspring = max(min(offspring, ub_rep), lb_rep);
end
% MATLAB Code
function [offspring] = updateFunc534(popdecs, popfits, cons)
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
    
    % Adaptive weights combining fitness and constraints
    w = 0.6 * norm_fit + 0.4 * norm_con;
    
    % Select elite individual (best feasible or least infeasible)
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        elite = popdecs(feasible_mask,:);
        elite = elite(elite_idx,:);
    else
        [~, elite_idx] = min(norm_fit + norm_con);
        elite = popdecs(elite_idx,:);
    end
    
    % Create feasible pool
    if any(feasible_mask)
        feasible_pop = popdecs(feasible_mask,:);
        feasible_idx = randi(size(feasible_pop,1), NP, 1);
        feasible_diff = feasible_pop(feasible_idx,:) - popdecs;
    else
        feasible_diff = zeros(NP, D);
    end
    
    % Generate random indices for diversity
    idx = 1:NP;
    r1 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx);
    r2 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx);
    rand_diff = popdecs(r1,:) - popdecs(r2,:);
    
    % Calculate elite direction
    elite_diff = repmat(elite, NP, 1) - popdecs;
    
    % Adaptive scaling factors
    F1 = 0.8 * (1 - w);
    F2 = 0.1 * w .* (1 - norm_con);
    F3 = 0.1 * norm_con;
    
    % Mutation operation
    offspring = popdecs + repmat(F1, 1, D) .* elite_diff + ...
                repmat(F2, 1, D) .* feasible_diff + ...
                repmat(F3, 1, D) .* rand_diff;
    
    % Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    mask_low = offspring < lb_rep;
    mask_high = offspring > ub_rep;
    offspring = offspring .* ~(mask_low | mask_high) + ...
                (2*lb_rep - offspring) .* mask_low + ...
                (2*ub_rep - offspring) .* mask_high;
    
    % Final boundary check
    offspring = max(min(offspring, ub_rep), lb_rep);
end
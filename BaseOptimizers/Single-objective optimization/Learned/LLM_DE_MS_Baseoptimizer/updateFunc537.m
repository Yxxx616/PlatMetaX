% MATLAB Code
function [offspring] = updateFunc537(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Normalize fitness (assuming minimization)
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fit = (popfits - f_min) / (f_max - f_min + eps);
    
    % Normalize constraint violations
    abs_cons = abs(cons);
    c_min = min(abs_cons);
    c_max = max(abs_cons);
    norm_con = (abs_cons - c_min) / (c_max - c_min + eps);
    
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
    
    % Generate random indices for difference vectors
    idx = 1:NP;
    r1 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    r2 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    
    % Adaptive scaling factors
    F1 = 0.6 * (1 - norm_fit);  % Fitness-adaptive elite guidance
    F2 = 0.4 + 0.3 * norm_con;   % Constraint-enhanced exploration
    F3 = 0.2 * norm_fit;         % Fitness-dependent perturbation
    
    % Mutation operation
    elite_diff = repmat(elite, NP, 1) - popdecs;
    rand_diff = popdecs(r1,:) - popdecs(r2,:);
    gauss_noise = randn(NP, D);
    
    offspring = popdecs + ...
        F1 .* elite_diff + ...
        F2 .* rand_diff .* repmat(norm_fit, 1, D) + ...
        F3 .* gauss_noise .* repmat(1 - norm_con, 1, D);
    
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
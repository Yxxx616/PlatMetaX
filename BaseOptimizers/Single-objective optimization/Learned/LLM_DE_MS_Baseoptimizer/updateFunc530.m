% MATLAB Code
function [offspring] = updateFunc530(popdecs, popfits, cons)
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
    
    % Adaptive weights (alpha=0.7)
    w = 0.7 * norm_fit + 0.3 * norm_con;
    
    % Select elite individual
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible,:);
        elite = elite(elite_idx,:);
    else
        [~, elite_idx] = min(popfits + 10*norm_con);
        elite = popdecs(elite_idx,:);
    end
    
    % Generate distinct random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx);
    r2 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx);
    r3 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx);
    r4 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx);
    
    % Calculate direction vectors
    diff1 = repmat(elite, NP, 1) - popdecs;
    diff2 = popdecs(r1,:) - popdecs(r2,:);
    diff3 = popdecs(r3,:) - popdecs(r4,:);
    
    % Adaptive scaling factors
    F1 = 0.8 * (1 - w);
    F2 = 0.6 * w;
    F3 = 0.4 * (1 - w).^2;
    
    % Mutation operation
    offspring = popdecs + repmat(F1, 1, D) .* diff1 + ...
                repmat(F2, 1, D) .* diff2 + ...
                repmat(F3, 1, D) .* diff3;
    
    % Boundary handling with adaptive perturbation
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Project to bounds
    offspring = max(min(offspring, ub_rep), lb_rep);
    
    % Add adaptive Gaussian perturbation
    perturb = randn(NP,D) .* repmat(0.05*w,1,D) .* (ub_rep-lb_rep);
    offspring = max(min(offspring + perturb, ub_rep), lb_rep);
end
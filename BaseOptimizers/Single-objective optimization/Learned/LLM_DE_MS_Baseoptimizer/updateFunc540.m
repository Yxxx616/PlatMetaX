% MATLAB Code
function [offspring] = updateFunc540(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Rank-based fitness weights (0=best, 1=worst)
    [~, fit_rank] = sort(popfits);
    w = (fit_rank - 1)' / (NP - 1 + eps);
    
    % Constraint violation weights (0=feasible, 1=most violated)
    abs_cons = abs(cons);
    v = (abs_cons - min(abs_cons)) / (max(abs_cons) - min(abs_cons) + eps);
    
    % Select elite individual (best feasible or least infeasible)
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        elite = popdecs(feasible_mask,:);
        elite = elite(elite_idx,:);
    else
        [~, elite_idx] = min(abs_cons);
        elite = popdecs(elite_idx,:);
    end
    
    % Generate random indices for difference vectors
    idx = 1:NP;
    r1 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    r2 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    
    % Adaptive scaling factors
    F1 = 0.8 * (1 - w);          % Elite guidance strength
    F2 = 0.5 * w + 0.3 * v;      % Constraint-aware perturbation
    F3 = 0.2 * (1 - v);           % Exploration control
    
    % Mutation operation
    elite_diff = repmat(elite, NP, 1) - popdecs;
    rand_diff = popdecs(r1,:) - popdecs(r2,:);
    gauss_noise = randn(NP, D);
    
    offspring = popdecs + ...
        F1 .* elite_diff + ...
        F2 .* rand_diff .* (1 + repmat(abs(cons), 1, D)) + ...
        F3 .* gauss_noise .* repmat(1 - w, 1, D);
    
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
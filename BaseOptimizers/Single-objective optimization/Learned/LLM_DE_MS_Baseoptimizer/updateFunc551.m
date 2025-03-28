% MATLAB Code
function [offspring] = updateFunc551(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Elite selection with constraint handling
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible(elite_idx),:);
    else
        [~, min_cons_idx] = min(abs(cons));
        elite = popdecs(min_cons_idx,:);
    end
    
    % Rank-based fitness weights
    [~, fit_rank] = sort(popfits);
    w_f = (fit_rank-1)'/(NP-1);  % Normalized [0,1]
    
    % Constraint violation weights
    abs_cons = abs(cons);
    cons_min = min(abs_cons);
    cons_max = max(abs_cons);
    w_c = (abs_cons - cons_min) ./ (cons_max - cons_min + eps);
    
    % Generate random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    r2 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    
    % Adaptive parameters
    F_e = 0.5 * (1 - w_f) .* (1 - w_c);
    F_r = 0.3 + 0.4 * w_f;
    sigma = 0.2 * (1 + w_c);
    
    % Mutation components
    elite_diff = repmat(elite, NP, 1) - popdecs;
    rand_diff = popdecs(r1,:) - popdecs(r2,:);
    perturbation = randn(NP, D) .* repmat(sigma, 1, D);
    
    % Combined mutation
    offspring = popdecs + ...
        repmat(F_e, 1, D) .* elite_diff + ...
        repmat(F_r, 1, D) .* rand_diff + ...
        perturbation;
    
    % Adaptive boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    mask_low = offspring < lb_rep;
    mask_high = offspring > ub_rep;
    
    beta = 0.3 * (1 - repmat(w_f', 1, D)) + 0.1;
    offspring = offspring .* ~(mask_low | mask_high) + ...
        (lb_rep + beta .* (popdecs - lb_rep)) .* mask_low + ...
        (ub_rep - beta .* (ub_rep - popdecs)) .* mask_high;
    
    % Final boundary enforcement
    offspring = max(min(offspring, ub_rep), lb_rep);
end
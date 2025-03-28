% MATLAB Code
function [offspring] = updateFunc549(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Elite selection
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible(elite_idx),:);
    else
        [~, min_cons_idx] = min(abs(cons));
        [~, min_fit_idx] = min(popfits);
        alpha = 0.6;
        elite = alpha*popdecs(min_cons_idx,:) + (1-alpha)*popdecs(min_fit_idx,:);
    end
    
    % Rank-based weights
    [~, fit_rank] = sort(popfits);
    w_f = (fit_rank-1)'/(NP-1);  % Normalized [0,1]
    
    % Constraint violation weights
    abs_cons = abs(cons);
    cons_min = min(abs_cons);
    cons_max = max(abs_cons);
    w_c = (abs_cons - cons_min) / (cons_max - cons_min + eps);
    
    % Generate random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    r2 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    
    % Mutation components
    F_e = 0.5 * (1 - w_f);
    F_c = 0.3 * w_c;
    F_f = 0.2 * (1 - w_c);
    sigma = 0.1 + 0.4 * w_f;
    
    elite_diff = repmat(elite, NP, 1) - popdecs;
    rand_diff = popdecs(r1,:) - popdecs(r2,:);
    perturbation = randn(NP, D) .* repmat(sigma, 1, D);
    
    offspring = popdecs + ...
        repmat(F_e, 1, D) .* elite_diff + ...
        repmat(F_c, 1, D) .* rand_diff + ...
        repmat(F_f, 1, D) .* perturbation;
    
    % Adaptive boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    mask_low = offspring < lb_rep;
    mask_high = offspring > ub_rep;
    
    beta = 0.3 + 0.5 * w_f';
    offspring = offspring .* ~(mask_low | mask_high) + ...
        (lb_rep + beta .* (popdecs - lb_rep)) .* mask_low + ...
        (ub_rep - beta .* (ub_rep - popdecs)) .* mask_high;
    
    % Final boundary enforcement
    offspring = max(min(offspring, ub_rep), lb_rep);
end
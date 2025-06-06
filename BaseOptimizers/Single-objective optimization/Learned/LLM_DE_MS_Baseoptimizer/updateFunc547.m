% MATLAB Code
function [offspring] = updateFunc547(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Select elite individual with adaptive strategy
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible(elite_idx),:);
    else
        % Combine best feasible attempt and best fitness
        [~, min_cons_idx] = min(abs(cons));
        [~, min_fit_idx] = min(popfits);
        elite = 0.7*popdecs(min_cons_idx,:) + 0.3*popdecs(min_fit_idx,:);
    end
    
    % Calculate fitness-based weights (normalized rank)
    [~, fit_rank] = sort(popfits);
    w = (fit_rank-1)'/(NP-1);  % Normalized to [0,1]
    
    % Calculate constraint violation weights
    abs_cons = abs(cons);
    cons_min = min(abs_cons);
    cons_max = max(abs_cons);
    v = (abs_cons - cons_min) / (cons_max - cons_min + eps);
    
    % Generate random indices without repetition
    idx = 1:NP;
    r1 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    r2 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    
    % Calculate adaptive parameters
    F_e = 0.8*(1-w) + 0.2*(1-v);
    F_c = 0.4*v;
    sigma = 0.1 + 0.3*w;
    
    % Mutation operation (vectorized)
    elite_diff = repmat(elite, NP, 1) - popdecs;
    rand_diff = popdecs(r1,:) - popdecs(r2,:);
    noise = randn(NP, D) .* repmat(sigma, 1, D);
    
    offspring = popdecs + ...
        repmat(F_e, 1, D) .* elite_diff + ...
        repmat(F_c, 1, D) .* rand_diff + ...
        noise;
    
    % Enhanced boundary handling with adaptive reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    mask_low = offspring < lb_rep;
    mask_high = offspring > ub_rep;
    
    alpha = 0.6 * (1 + w');
    beta = 0.4 * (1 - w');
    
    offspring = offspring .* ~(mask_low | mask_high) + ...
        (alpha .* lb_rep + beta .* popdecs) .* mask_low + ...
        (alpha .* ub_rep + beta .* popdecs) .* mask_high;
    
    % Final boundary enforcement
    offspring = max(min(offspring, ub_rep), lb_rep);
end
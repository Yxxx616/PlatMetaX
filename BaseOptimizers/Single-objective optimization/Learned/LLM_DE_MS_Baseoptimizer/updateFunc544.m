% MATLAB Code
function [offspring] = updateFunc544(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Select elite individual
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible,:);
        elite = elite(elite_idx,:);
    else
        [~, elite_idx] = min(abs(cons));
        elite = popdecs(elite_idx,:);
    end
    
    % Calculate fitness weights
    [~, fit_rank] = sort(popfits);
    w = fit_rank' / NP;
    
    % Calculate constraint violation weights
    abs_cons = abs(cons);
    v = (abs_cons - min(abs_cons)) / (max(abs_cons) - min(abs_cons) + eps);
    
    % Generate random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    r2 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    
    % Calculate adaptive parameters
    F_e = 0.8 * (1 - w) + 0.2 * (1 - v);
    F_c = 0.4 * v;
    sigma = 0.1 * (1 - w);
    
    % Mutation operation
    elite_diff = repmat(elite, NP, 1) - popdecs;
    rand_diff = popdecs(r1,:) - popdecs(r2,:);
    noise = randn(NP, D) .* repmat(sigma, 1, D);
    
    offspring = popdecs + ...
        repmat(F_e, 1, D) .* elite_diff + ...
        repmat(F_c, 1, D) .* rand_diff + ...
        noise;
    
    % Boundary handling with midpoint reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    mask_low = offspring < lb_rep;
    mask_high = offspring > ub_rep;
    
    offspring = offspring .* ~(mask_low | mask_high) + ...
        ((lb_rep + popdecs)/2) .* mask_low + ...
        ((ub_rep + popdecs)/2) .* mask_high;
    
    % Final boundary enforcement
    offspring = max(min(offspring, ub_rep), lb_rep);
end
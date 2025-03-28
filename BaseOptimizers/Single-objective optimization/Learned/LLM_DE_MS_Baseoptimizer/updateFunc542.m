% MATLAB Code
function [offspring] = updateFunc542(popdecs, popfits, cons)
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
    
    % Calculate weights
    [~, fit_rank] = sort(popfits);
    w = (fit_rank - 1)' / (NP - 1 + eps);
    
    abs_cons = abs(cons);
    v = (abs_cons - min(abs_cons)) / (max(abs_cons) - min(abs_cons) + eps);
    
    % Generate random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    r2 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    
    % Calculate scaling factors
    F_base = 0.5 + 0.3 * cos(pi * w');
    F_con = 0.3 * (1 - v);
    
    % Mutation operation
    elite_diff = repmat(elite, NP, 1) - popdecs;
    rand_diff = popdecs(r1,:) - popdecs(r2,:);
    noise = randn(NP, D) .* repmat(0.1*(1-w'), 1, D);
    
    offspring = popdecs + ...
        repmat(F_base, 1, D) .* elite_diff + ...
        repmat(F_con, 1, D) .* rand_diff + ...
        noise;
    
    % Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    mask_low = offspring < lb_rep;
    mask_high = offspring > ub_rep;
    
    offspring = offspring .* ~(mask_low | mask_high) + ...
        (2*lb_rep - offspring) .* mask_low + ...
        (2*ub_rep - offspring) .* mask_high;
    
    % Final boundary enforcement
    offspring = max(min(offspring, ub_rep), lb_rep);
end
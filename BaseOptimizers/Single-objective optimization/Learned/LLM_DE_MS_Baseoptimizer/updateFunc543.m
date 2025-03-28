% MATLAB Code
function [offspring] = updateFunc543(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Select elite individual based on feasibility
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible,:);
        elite = elite(elite_idx,:);
    else
        [~, elite_idx] = min(abs(cons));
        elite = popdecs(elite_idx,:);
    end
    
    % Calculate fitness-based weights
    [~, fit_rank] = sort(popfits);
    w = (fit_rank - 1)' / (NP - 1 + eps);
    
    % Calculate constraint violation weights
    abs_cons = abs(cons);
    v = (abs_cons - min(abs_cons)) / (max(abs_cons) - min(abs_cons) + eps);
    
    % Generate random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    r2 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    
    % Calculate scaling factors
    F_e = 0.7 * (1 - w') + 0.3 * (1 - v);
    F_c = 0.5 * v;
    sigma = 0.2 * (1 - w');
    
    % Mutation operation
    elite_diff = repmat(elite, NP, 1) - popdecs;
    rand_diff = popdecs(r1,:) - popdecs(r2,:);
    noise = randn(NP, D) .* repmat(sigma, 1, D);
    
    offspring = popdecs + ...
        repmat(F_e, 1, D) .* elite_diff + ...
        repmat(F_c, 1, D) .* rand_diff + ...
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
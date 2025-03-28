% MATLAB Code
function [offspring] = updateFunc565(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Elite selection with constraint handling
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible(elite_idx), :);
    else
        [~, min_cons_idx] = min(cons);
        elite = popdecs(min_cons_idx, :);
    end
    
    % Calculate ranks and normalized constraints
    [~, rank_fit] = sort(popfits);
    norm_cons = abs(cons) ./ (max(abs(cons)) + eps);
    
    % Generate random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    r2 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    
    % Adaptive scaling factors
    F = 0.5 + 0.3 * (1 - rank_fit/NP) .* (1 - norm_cons);
    
    % Mutation
    elite_diff = repmat(elite, NP, 1) - popdecs;
    rand_diff = popdecs(r1,:) - popdecs(r2,:);
    perturbation = 0.1 * randn(NP, D);
    
    offspring = popdecs + bsxfun(@times, F, elite_diff) + ...
                bsxfun(@times, F, rand_diff) + perturbation;
    
    % Boundary handling with midpoint reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    mask_low = offspring < lb_rep;
    mask_high = offspring > ub_rep;
    
    offspring = offspring .* ~(mask_low | mask_high) + ...
                0.5*(popdecs + lb_rep) .* mask_low + ...
                0.5*(popdecs + ub_rep) .* mask_high;
    
    % Final boundary enforcement
    offspring = max(min(offspring, ub_rep), lb_rep);
end
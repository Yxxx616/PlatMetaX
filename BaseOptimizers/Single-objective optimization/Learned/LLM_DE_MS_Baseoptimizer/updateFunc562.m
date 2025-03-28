% MATLAB Code
function [offspring] = updateFunc562(popdecs, popfits, cons)
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
    
    % Feasibility ratio
    rho = sum(feasible) / NP;
    
    % Normalized constraint violation
    norm_cons = abs(cons) ./ (max(abs(cons)) + eps);
    
    % Adaptive scaling factors
    F = 0.5 * (1 + rho) .* (1 - norm_cons);
    
    % Generate four distinct random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    r2 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    r3 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    r4 = arrayfun(@(i) setdiff(idx, i)(randi(NP-1)), idx)';
    
    % Mutation components
    elite_diff = repmat(elite, NP, 1) - popdecs;
    rand_diff = popdecs(r1,:) - popdecs(r2,:) + popdecs(r3,:) - popdecs(r4,:);
    perturbation = 0.1 * randn(NP, D) .* (1 - rho) .* sqrt(sum(elite_diff.^2, 2));
    
    % Combined mutation
    offspring = popdecs + ...
        repmat(F, 1, D) .* elite_diff + ...
        0.5 * (1 - rho) * rand_diff + ...
        perturbation;
    
    % Boundary handling with repair
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Hard boundary enforcement first
    offspring = max(min(offspring, ub_rep), lb_rep);
    
    % Repair strategy for boundary violations
    mask_low = offspring <= lb_rep;
    mask_high = offspring >= ub_rep;
    
    offspring = offspring .* ~(mask_low | mask_high) + ...
        (popdecs + 0.5 * (lb_rep - popdecs)) .* mask_low + ...
        (popdecs + 0.5 * (ub_rep - popdecs)) .* mask_high;
    
    % Final boundary check
    offspring = max(min(offspring, ub_rep), lb_rep);
end
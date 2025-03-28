% MATLAB Code
function [offspring] = updateFunc448(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Identify elite solution (best feasible or least infeasible)
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        elite = popdecs(feasible_mask(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % Calculate feasible center and infeasible center
    if any(feasible_mask)
        feas_center = mean(popdecs(feasible_mask, :), 1);
        infeas_center = mean(popdecs(~feasible_mask, :), 1);
    else
        feas_center = mean(popdecs, 1);
        infeas_center = feas_center;
    end
    
    % Adaptive scaling factors based on normalized fitness and constraints
    norm_fits = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    norm_cons = (cons - min(cons)) / (max(cons) - min(cons) + eps);
    
    F1 = 0.5 * (1 + norm_fits);
    F2 = 0.5 * (1 - norm_cons);
    F3 = 0.3 * (1 + norm_cons);
    
    % Expand to D dimensions
    F1 = repmat(F1, 1, D);
    F2 = repmat(F2, 1, D);
    F3 = repmat(F3, 1, D);
    
    % Generate random indices (vectorized)
    r1 = zeros(NP, 1);
    r2 = zeros(NP, 1);
    for i = 1:NP
        available = setdiff(1:NP, i);
        r1(i) = available(randi(NP-1));
        available = setdiff(available, r1(i));
        r2(i) = available(randi(NP-2));
    end
    
    % Direction vectors
    elite_dir = repmat(elite, NP, 1) - popdecs;
    feas_dir = repmat(feas_center, NP, 1) - popdecs;
    div_dir = popdecs(r1, :) - popdecs(r2, :);
    
    % Combined mutation with adaptive weights
    offspring = popdecs + F1.*elite_dir + F2.*feas_dir + F3.*div_dir;
    
    % Boundary handling with reflection and clipping
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Reflection for out-of-bound solutions
    mask_low = offspring < lb_rep;
    mask_high = offspring > ub_rep;
    
    offspring = offspring .* ~(mask_low | mask_high) + ...
        (2*lb_rep - offspring) .* mask_low + ...
        (2*ub_rep - offspring) .* mask_high;
    
    % Final clipping to ensure strict bounds
    offspring = max(min(offspring, ub_rep), lb_rep);
    
    % Additional perturbation for diversity
    perturb_mask = rand(NP, D) < 0.1;
    offspring = offspring + perturb_mask .* randn(NP, D) .* 0.1 .* (ub_rep - lb_rep);
    offspring = max(min(offspring, ub_rep), lb_rep);
end
% MATLAB Code
function [offspring] = updateFunc445(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Feasibility information
    feasible_mask = cons <= 0;
    feasible_count = sum(feasible_mask);
    
    % Identify elite solution
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        elite = popdecs(feasible_mask(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % Calculate feasible center
    if any(feasible_mask)
        feas_center = mean(popdecs(feasible_mask, :), 1);
    else
        feas_center = mean(popdecs, 1);
    end
    
    % Normalize constraints and fitness
    min_cons = min(cons);
    max_cons = max(cons);
    norm_cons = (cons - min_cons) / (max_cons - min_cons + eps);
    
    min_fit = min(popfits);
    max_fit = max(popfits);
    norm_fits = (popfits - min_fit) / (max_fit - min_fit + eps);
    
    % Adaptive parameters
    F = 0.5 * (1 + norm_cons);
    lambda = 0.5 * (1 - norm_fits);
    
    % Expand parameters to D dimensions
    F = repmat(F, 1, D);
    lambda = repmat(lambda, 1, D);
    
    % Generate random indices
    idx = 1:NP;
    r1 = zeros(NP,1);
    r2 = zeros(NP,1);
    for i = 1:NP
        available = idx(idx ~= i);
        r1(i) = available(randi(NP-1));
        available = available(available ~= r1(i));
        r2(i) = available(randi(NP-2));
    end
    
    % Direction vectors
    feas_dir = repmat(feas_center, NP, 1) - popdecs;
    elite_dir = repmat(elite, NP, 1) - popdecs;
    div_dir = popdecs(r1, :) - popdecs(r2, :);
    
    % Combined mutation
    offspring = popdecs + F.*(elite_dir + feas_dir) + lambda.*div_dir;
    
    % Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    mask_low = offspring < lb_rep;
    mask_high = offspring > ub_rep;
    
    offspring = offspring .* ~(mask_low | mask_high) + ...
        (2*lb_rep - offspring) .* mask_low + ...
        (2*ub_rep - offspring) .* mask_high;
    
    % Final clipping to ensure strict bounds
    offspring = max(min(offspring, ub_rep), lb_rep);
end
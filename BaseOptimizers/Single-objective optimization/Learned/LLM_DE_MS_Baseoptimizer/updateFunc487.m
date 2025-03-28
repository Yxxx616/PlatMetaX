% MATLAB Code
function [offspring] = updateFunc487(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Elite selection based on feasibility
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        elite = popdecs(feasible_mask(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % Normalize constraint violations
    min_cv = min(cons);
    max_cv = max(cons);
    norm_cv = (cons - min_cv) / (max_cv - min_cv + eps);
    
    % Adaptive scaling factors based on constraints
    F = 0.5 * (1 + norm_cv);
    
    % Generate random indices (avoiding current index)
    idx = arrayfun(@(x) setdiff(randperm(NP), x), 1:NP, 'UniformOutput', false);
    r1 = cellfun(@(x) x(1), idx)';
    r2 = cellfun(@(x) x(2), idx)';
    r3 = cellfun(@(x) x(3), idx)';
    r4 = cellfun(@(x) x(4), idx)';
    
    % Constraint-aware mutation
    elite_dir = repmat(elite, NP, 1) - popdecs;
    diff_vec1 = popdecs(r1,:) - popdecs(r2,:);
    diff_vec2 = popdecs(r3,:) - popdecs(r4,:);
    
    offspring = popdecs + repmat(F, 1, D).*(elite_dir + diff_vec1) + ...
                repmat(1-F, 1, D).*diff_vec2;
    
    % Boundary handling with midpoint reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    mask_low = offspring < lb_rep;
    mask_high = offspring > ub_rep;
    
    offspring = offspring.*(~mask_low & ~mask_high) + ...
               ((popdecs + lb_rep)/2).*mask_low + ...
               ((popdecs + ub_rep)/2).*mask_high;
    
    % Final clipping for numerical stability
    offspring = max(min(offspring, ub_rep), lb_rep);
end
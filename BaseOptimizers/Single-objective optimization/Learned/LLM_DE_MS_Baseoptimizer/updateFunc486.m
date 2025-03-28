% MATLAB Code
function [offspring] = updateFunc486(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Feasibility ratio
    feasible_mask = cons <= 0;
    alpha = sum(feasible_mask) / NP;
    
    % Elite selection
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        elite = popdecs(feasible_mask(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % Normalize fitness
    min_fit = min(popfits);
    max_fit = max(popfits);
    norm_fits = (popfits - min_fit) / (max_fit - min_fit + eps);
    
    % Adaptive scaling factors
    F = 0.4 + 0.6 * norm_fits;
    
    % Generate random indices (avoiding current index)
    idx = arrayfun(@(x) setdiff(randperm(NP), x), 1:NP, 'UniformOutput', false);
    r1 = cellfun(@(x) x(1), idx)';
    r2 = cellfun(@(x) x(2), idx)';
    r3 = cellfun(@(x) x(3), idx)';
    r4 = cellfun(@(x) x(4), idx)';
    
    % Mutation vectors
    elite_dir = repmat(elite, NP, 1) - popdecs;
    diff_vec = alpha*(popdecs(r1,:) - popdecs(r2,:)) + ...
              (1-alpha)*(popdecs(r3,:) - popdecs(r4,:));
    
    % Generate offspring
    offspring = popdecs + repmat(F, 1, D).*(elite_dir + diff_vec);
    
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
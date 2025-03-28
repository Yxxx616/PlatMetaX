% MATLAB Code
function [offspring] = updateFunc485(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Feasibility analysis
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
    
    % Normalize fitness and constraints
    norm_fits = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    norm_cons = (cons - min(cons)) / (max(cons) - min(cons) + eps);
    
    % Adaptive scaling factors
    F = 0.5 * (1 + norm_fits);
    epsilon = 0.1 * (1 - norm_cons);
    
    % Generate random indices
    idx = arrayfun(@(x) randperm(NP-1, 4), 1:NP, 'UniformOutput', false);
    r1 = cellfun(@(x) x(1), idx)';
    r2 = cellfun(@(x) x(2), idx)';
    r3 = cellfun(@(x) x(3), idx)';
    r4 = cellfun(@(x) x(4), idx)';
    
    % Feasibility-weighted difference vectors
    diff_feas = alpha*(popdecs(r1,:) - popdecs(r2,:)) + ...
               (1-alpha)*(popdecs(r3,:) - popdecs(r4,:));
    
    % Elite direction vectors
    elite_dir = repmat(elite, NP, 1) - popdecs;
    
    % Mutation with adaptive scaling
    offspring = popdecs + repmat(F, 1, D).*(elite_dir + diff_feas) + ...
               repmat(epsilon, 1, D).*randn(NP, D);
    
    % Boundary handling - midpoint reflection
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
% MATLAB Code
function [offspring] = updateFunc489(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Identify elite solution
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        elite = popdecs(feasible_mask, :);
        elite = elite(elite_idx, :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % Normalize fitness and constraints
    min_fit = min(popfits);
    max_fit = max(popfits);
    norm_fits = (popfits - min_fit) / (max_fit - min_fit + eps);
    
    max_cv = max(cons);
    norm_cv = cons / (max_cv + eps);
    
    % Calculate scaling factors
    F1 = 0.7 * (1 - norm_cv);
    F2 = 0.5 * norm_fits;
    F3 = 0.3 * norm_cv;
    
    % Generate random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    while any(r1 == r2)
        r2 = randi(NP, NP, 1);
    end
    
    % Mutation operation
    elite_dir = repmat(elite, NP, 1) - popdecs;
    diff_feasible = popdecs(r1,:) - popdecs(r2,:);
    rand_comp = randn(NP, D);
    
    offspring = popdecs + F1.*elite_dir + F2.*diff_feasible + F3.*rand_comp;
    
    % Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Reflect back into bounds if out of bounds
    mask_low = offspring < lb_rep;
    mask_high = offspring > ub_rep;
    offspring = offspring .* ~(mask_low | mask_high) + ...
               (2*lb_rep - offspring) .* mask_low + ...
               (2*ub_rep - offspring) .* mask_high;
    
    % Final clipping to ensure within bounds
    offspring = max(min(offspring, ub_rep), lb_rep);
end
% MATLAB Code
function [offspring] = updateFunc233(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Boundary definitions
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize constraints
    abs_cons = abs(cons);
    c_min = min(abs_cons);
    c_max = max(abs_cons) + eps;
    c_weights = 1 ./ (1 + exp(-5 * (abs_cons - c_min) ./ (c_max - c_min + eps));
    
    % Normalize fitness
    f_min = min(popfits);
    f_max = max(popfits) + eps;
    f_weights = 1 ./ (1 + exp(5 * (popfits - f_min) ./ (f_max - f_min + eps)));
    
    % Select elite individual (best feasible or least infeasible)
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        elite = popdecs(feasible_mask(elite_idx), :);
    else
        [~, elite_idx] = min(abs_cons);
        elite = popdecs(elite_idx, :);
    end
    
    % Select best feasible individual
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
        temp = popdecs(feasible_mask, :);
        best_feas = temp(best_idx, :);
    else
        best_feas = elite;
    end
    
    % Pre-compute random pairs
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    same_idx = (r1 == (1:NP)') | (r2 == (1:NP)') | (r1 == r2);
    while any(same_idx)
        r1(same_idx) = randi(NP, sum(same_idx), 1);
        r2(same_idx) = randi(NP, sum(same_idx), 1);
        same_idx = (r1 == (1:NP)') | (r2 == (1:NP)') | (r1 == r2);
    end
    
    % Vectorized mutation
    d_elite = repmat(elite, NP, 1) - popdecs;
    d_feas = repmat(best_feas, NP, 1) - popdecs;
    d_rand = popdecs(r1,:) - popdecs(r2,:);
    
    w3 = 0.1 + 0.1 * rand(NP, 1);
    offspring = popdecs + f_weights.*d_elite + c_weights.*d_feas + w3.*d_rand;
    
    % Boundary handling with reflection
    out_low = offspring < lb;
    out_high = offspring > ub;
    
    offspring = offspring.*(~out_low & ~out_high) + ...
               (lb + (lb - offspring)).*out_low + ...
               (ub - (offspring - ub)).*out_high;
    
    % Final boundary enforcement
    offspring = max(min(offspring, ub), lb);
end
% MATLAB Code
function [offspring] = updateFunc251(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Boundary definitions
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Process constraints (0 for feasible, positive for infeasible)
    abs_cons = max(0, cons);
    c_min = min(abs_cons);
    c_max = max(abs_cons) + eps;
    
    % Process fitness (assuming minimization)
    f_min = min(popfits);
    f_max = max(popfits) + eps;
    
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
    
    % Generate random pairs ensuring they're distinct
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    same_idx = (r1 == (1:NP)') | (r2 == (1:NP)') | (r1 == r2);
    while any(same_idx)
        r1(same_idx) = randi(NP, sum(same_idx), 1);
        r2(same_idx) = randi(NP, sum(same_idx), 1);
        same_idx = (r1 == (1:NP)') | (r2 == (1:NP)') | (r1 == r2);
    end
    
    % Calculate adaptive weights with enhanced exploration
    f_weights = 0.5 * (1 - (popfits - f_min) ./ (f_max - f_min)) + 0.1 * rand(NP, 1);
    c_weights = 0.3 * (1 - (abs_cons - c_min) ./ (c_max - c_min)) + 0.1 * rand(NP, 1);
    r_weights = 0.2 * (abs_cons ./ c_max) .* rand(NP, 1);
    
    % Vectorized mutation operation with enhanced exploration
    d_elite = repmat(elite, NP, 1) - popdecs;
    d_feas = repmat(best_feas, NP, 1) - popdecs;
    d_rand = popdecs(r1,:) - popdecs(r2,:);
    
    % Enhanced mutation with additional random component
    offspring = popdecs + f_weights.*d_elite + c_weights.*d_feas + r_weights.*d_rand + ...
                0.05 * randn(NP, D) .* (ub - lb);
    
    % Enhanced boundary handling with reflection and randomization
    out_low = offspring < lb;
    out_high = offspring > ub;
    
    reflect_low = lb + rand(NP,D).*(popdecs - lb);
    reflect_high = ub - rand(NP,D).*(ub - popdecs);
    
    offspring = offspring.*(~out_low & ~out_high) + ...
               reflect_low.*out_low + reflect_high.*out_high;
    
    % Final boundary enforcement with small perturbation
    offspring = max(min(offspring, ub), lb) + 0.01 * randn(NP, D) .* (ub - lb);
    offspring = max(min(offspring, ub), lb);
end
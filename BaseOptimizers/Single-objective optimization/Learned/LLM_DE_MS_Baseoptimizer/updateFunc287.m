% MATLAB Code
function [offspring] = updateFunc287(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Boundary definitions
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Process constraints
    abs_cons = max(0, cons);
    c_min = min(abs_cons);
    c_max = max(abs_cons) + eps;
    
    % Process fitness (minimization)
    f_min = min(popfits);
    f_max = max(popfits) + eps;
    
    % Select elite individual (best feasible or least infeasible)
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        temp = popdecs(feasible_mask, :);
        elite = temp(elite_idx, :);
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
    
    % Select individual with lowest constraint violation
    [~, lowcons_idx] = min(abs_cons);
    lowcons = popdecs(lowcons_idx, :);
    
    % Generate random pairs ensuring distinct indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    same_idx = (r1 == (1:NP)') | (r2 == (1:NP)') | (r1 == r2);
    while any(same_idx)
        r1(same_idx) = randi(NP, sum(same_idx), 1);
        r2(same_idx) = randi(NP, sum(same_idx), 1);
        same_idx = (r1 == (1:NP)') | (r2 == (1:NP)') | (r1 == r2);
    end
    
    % Calculate adaptive weights
    f_weights = 0.4 + 0.4*(popfits - f_min) ./ (f_max - f_min);
    c_weights = 0.4 + 0.4*(abs_cons - c_min) ./ (c_max - c_min);
    F = 0.6 + 0.2*sin(2*pi*rand(NP,1)); % Cyclic scaling factor
    
    % Vectorized mutation components
    d_elite = repmat(elite, NP, 1) - popdecs;
    d_feas = repmat(best_feas, NP, 1) - popdecs;
    d_rand = popdecs(r1,:) - popdecs(r2,:);
    d_cons = repmat(lowcons, NP, 1) - popdecs;
    d_opp = repmat(lb + ub, NP, 1) - popdecs;
    
    % Adaptive mutation based on feasibility
    feasible_terms = F.*d_elite + f_weights.*d_feas + 0.9*d_rand;
    infeasible_terms = F.*d_elite + c_weights.*d_cons + 0.5*d_opp;
    
    offspring = popdecs + feasible_terms.*repmat(feasible_mask,1,D) + ...
                infeasible_terms.*repmat(~feasible_mask,1,D);
    
    % Boundary handling with adaptive reflection
    out_low = offspring < lb;
    out_high = offspring > ub;
    eta = 0.15 + 0.3*rand(NP,D); % Random perturbation factor
    
    offspring = offspring.*(~out_low & ~out_high) + ...
               (lb + eta.*(ub - offspring)).*out_low + ...
               (ub - eta.*(offspring - lb)).*out_high;
    
    % Final boundary enforcement
    offspring = max(min(offspring, ub), lb);
end
% MATLAB Code
function [offspring] = updateFunc288(popdecs, popfits, cons)
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
    
    % Rank individuals based on fitness and constraints
    [~, f_rank] = sort(popfits);
    [~, c_rank] = sort(abs_cons);
    
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
    
    % Calculate adaptive weights and scaling factors
    F = 0.5 + 0.3*sin(pi*f_rank/NP); % Cyclic scaling based on rank
    w1 = 0.4 + 0.3*rand(NP,1); % Elite weight
    w2 = 0.3 + 0.2*rand(NP,1); % Feasible weight
    w3 = 0.2 + 0.1*rand(NP,1); % Random weight
    w4 = 0.1 + 0.1*rand(NP,1); % Constraint weight
    
    % Vectorized mutation components
    d_elite = repmat(elite, NP, 1) - popdecs;
    d_feas = repmat(best_feas, NP, 1) - popdecs;
    d_rand = popdecs(r1,:) - popdecs(r2,:);
    d_cons = repmat(lowcons, NP, 1) - popdecs;
    
    % Adaptive mutation
    offspring = popdecs + repmat(F,1,D).* ...
               (repmat(w1,1,D).*d_elite + repmat(w2,1,D).*d_feas + ...
                repmat(w3,1,D).*d_rand + repmat(w4,1,D).*d_cons);
    
    % Boundary handling with adaptive reflection
    out_low = offspring < lb;
    out_high = offspring > ub;
    eta = 0.2 + 0.3*rand(NP,D); % Random perturbation factor
    
    offspring = offspring.*(~out_low & ~out_high) + ...
               (lb + eta.*(ub - offspring)).*out_low + ...
               (ub - eta.*(offspring - lb)).*out_high;
    
    % Final boundary enforcement
    offspring = max(min(offspring, ub), lb);
end
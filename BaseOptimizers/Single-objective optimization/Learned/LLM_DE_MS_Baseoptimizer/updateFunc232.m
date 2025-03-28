% MATLAB Code
function [offspring] = updateFunc232(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Boundary definitions
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize constraints
    abs_cons = abs(cons);
    c_min = min(abs_cons);
    c_max = max(abs_cons) + eps;
    c_weights = 1 ./ (1 + exp(-5 * (abs_cons - c_min) ./ (c_max - c_min + eps)));
    
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
    
    % Vectorized mutation with adaptive weights
    for i = 1:NP
        % Select two random distinct individuals
        idxs = randperm(NP, 2);
        while any(idxs == i)
            idxs = randperm(NP, 2);
        end
        
        % Calculate direction vectors
        d_elite = elite - popdecs(i,:);
        d_feas = best_feas - popdecs(i,:);
        d_rand = popdecs(idxs(1),:) - popdecs(idxs(2),:);
        
        % Adaptive weights with random diversity component
        w1 = f_weights(i);
        w2 = c_weights(i);
        w3 = 0.1 + 0.1 * rand();
        
        % Combine components
        offspring(i,:) = popdecs(i,:) + w1.*d_elite + w2.*d_feas + w3.*d_rand;
    end
    
    % Boundary handling with reflection
    out_low = offspring < lb;
    out_high = offspring > ub;
    
    offspring = offspring.*(~out_low & ~out_high) + ...
               (lb + (lb - offspring)).*out_low + ...
               (ub - (offspring - ub)).*out_high;
    
    % Final boundary enforcement
    offspring = max(min(offspring, ub), lb);
end
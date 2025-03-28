% MATLAB Code
function [offspring] = updateFunc231(popdecs, popfits, cons)
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
    
    % Select elite individual
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
    
    % Vectorized mutation with random diversity component
    for i = 1:NP
        % Randomly select two distinct individuals
        idxs = randperm(NP, 2);
        while any(idxs == i)
            idxs = randperm(NP, 2);
        end
        
        % Calculate direction vectors
        d1 = elite - popdecs(i,:);
        d2 = best_feas - popdecs(i,:);
        d3 = popdecs(idxs(1),:) - popdecs(idxs(2),:);
        
        % Adaptive weights
        w1 = f_weights(i);
        w2 = c_weights(i);
        w3 = 0.2;  % Fixed weight for diversity
        
        % Combine components
        offspring(i,:) = popdecs(i,:) + w1.*d1 + w2.*d2 + w3.*d3;
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
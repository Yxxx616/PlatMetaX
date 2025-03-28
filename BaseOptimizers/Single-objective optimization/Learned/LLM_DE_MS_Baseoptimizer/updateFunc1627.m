% MATLAB Code
function [offspring] = updateFunc1627(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Create elite pool (top 30% considering both fitness and constraints)
    weighted_fits = popfits + 1e6 * max(0, cons);
    [~, sorted_idx] = sort(weighted_fits);
    elite_size = max(3, ceil(NP*0.3));
    elite_pool = popdecs(sorted_idx(1:elite_size), :);
    
    % 2. Compute adaptive parameters
    [~, rank_order] = sort(popfits);
    ranks = zeros(NP, 1);
    ranks(rank_order) = (1:NP)';
    
    c_abs = abs(cons);
    c_max = max(c_abs) + eps;
    
    % Adaptive scaling factor
    F = 0.5 * (1 + ranks./NP) .* (1 - c_abs./c_max);
    
    % Dynamic crossover rate
    CR = 0.9 * (1 - ranks./NP) .* (1 - c_abs./c_max);
    
    % 3. Generate offspring (vectorized)
    offspring = zeros(NP, D);
    for i = 1:NP
        % Select elite from pool
        elite = elite_pool(randi(elite_size), :);
        
        % Select 3 distinct random indices
        candidates = setdiff(1:NP, i);
        idx = candidates(randperm(length(candidates), 3));
        r1 = idx(1); r2 = idx(2); r3 = idx(3);
        
        % Enhanced mutation with elite guidance
        mutation = elite + F(i) * (popdecs(r1,:) - popdecs(r2,:)) + ...
                   F(i) * (popdecs(r3,:) - elite);
        
        % Adaptive crossover
        mask = rand(1, D) < CR(i);
        j_rand = randi(D);
        mask(j_rand) = true;
        
        offspring(i,:) = popdecs(i,:) .* (~mask) + mutation .* mask;
    end
    
    % 4. Enhanced boundary handling with midpoint reflection
    for j = 1:D
        viol_low = offspring(:,j) < lb(j);
        offspring(viol_low,j) = (lb(j) + popdecs(viol_low,j)) / 2;
        
        viol_high = offspring(:,j) > ub(j);
        offspring(viol_high,j) = (ub(j) + popdecs(viol_high,j)) / 2;
    end
    
    % Final bounds check
    offspring = max(min(offspring, ub), lb);
end
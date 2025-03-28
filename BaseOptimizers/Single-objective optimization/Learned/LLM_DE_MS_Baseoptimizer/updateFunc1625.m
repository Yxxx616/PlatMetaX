% MATLAB Code
function [offspring] = updateFunc1625(popdecs, popfits, cons)
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
    F = 0.3 + 0.5 * (1 - ranks./NP) .* (1 - c_abs./c_max);
    
    % Dynamic crossover rate
    CR = 0.9 * (1 - c_abs./c_max) .* (1 - ranks./NP);
    
    % 3. Generate offspring (vectorized)
    offspring = zeros(NP, D);
    for i = 1:NP
        % Select elite from pool
        elite = elite_pool(randi(elite_size), :);
        
        % Select 2 distinct random indices
        candidates = setdiff(1:NP, i);
        idx = candidates(randperm(length(candidates), 2));
        r1 = idx(1); r2 = idx(2);
        
        % Mutation with elite guidance
        mutation = popdecs(i,:) + F(i) * (elite - popdecs(i,:)) + F(i) * (popdecs(r1,:) - popdecs(r2,:));
        
        % Adaptive crossover
        mask = rand(1, D) < CR(i);
        j_rand = randi(D);
        mask(j_rand) = true;
        
        offspring(i,:) = popdecs(i,:) .* (~mask) + mutation .* mask;
    end
    
    % 4. Enhanced boundary handling with adaptive reflection
    for j = 1:D
        % Reflection with random factor
        viol_low = offspring(:,j) < lb(j);
        offspring(viol_low,j) = lb(j) + rand(sum(viol_low),1) .* (ub(j) - lb(j));
        
        viol_high = offspring(:,j) > ub(j);
        offspring(viol_high,j) = ub(j) - rand(sum(viol_high),1) .* (ub(j) - lb(j));
    end
    
    % Final bounds check
    offspring = max(min(offspring, ub), lb);
end
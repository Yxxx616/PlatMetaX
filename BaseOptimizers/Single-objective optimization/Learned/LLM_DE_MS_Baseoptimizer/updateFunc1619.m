% MATLAB Code
function [offspring] = updateFunc1619(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Create elite pool (top 20% considering both fitness and constraints)
    weighted_fits = popfits + 1e6 * max(0, cons);
    [~, sorted_idx] = sort(weighted_fits);
    elite_size = max(2, ceil(NP*0.2));
    elite_pool = popdecs(sorted_idx(1:elite_size), :);
    
    % 2. Compute adaptive parameters
    [~, rank_order] = sort(popfits);
    ranks = zeros(NP, 1);
    ranks(rank_order) = (1:NP)';
    F = 0.5 + 0.3 * (1 - ranks ./ NP);  % Fitness-ranked scaling
    
    c_abs = abs(cons);
    c_max = max(c_abs) + eps;
    CR = 0.9 * (1 - c_abs ./ c_max);    % Constraint-aware crossover
    
    % 3. Generate offspring (vectorized)
    offspring = zeros(NP, D);
    for i = 1:NP
        % Select elite from pool
        elite = elite_pool(randi(elite_size), :);
        elite_dir = elite - popdecs(i,:);
        
        % Select 2 distinct random indices
        candidates = setdiff(1:NP, i);
        idx = candidates(randperm(length(candidates), 2));
        r1 = idx(1); r2 = idx(2);
        
        % Constraint-aware perturbation
        pert_factor = 0.6 - 0.4 * (c_abs(i) / c_max);
        perturbation = pert_factor * (popdecs(r1,:) - popdecs(r2,:)) .* (1 + c_abs(i)/c_max);
        
        % Mutation with fitness-ranked scaling
        mutation = popdecs(i,:) + F(i) * elite_dir + perturbation + ...
                  0.2 * (1 - ranks(i)/NP) * randn(1, D);
        
        % Adaptive crossover
        mask = rand(1, D) < CR(i);
        j_rand = randi(D);
        mask(j_rand) = true;
        
        offspring(i,:) = popdecs(i,:) .* (~mask) + mutation .* mask;
    end
    
    % 4. Boundary handling with reflection and randomization
    for j = 1:D
        % Reflection for lower bound violations
        viol_low = offspring(:,j) < lb(j);
        offspring(viol_low,j) = 2*lb(j) - offspring(viol_low,j);
        
        % Reflection for upper bound violations
        viol_high = offspring(:,j) > ub(j);
        offspring(viol_high,j) = 2*ub(j) - offspring(viol_high,j);
        
        % Randomization for remaining violations
        still_viol = (offspring(:,j) < lb(j)) | (offspring(:,j) > ub(j));
        offspring(still_viol,j) = lb(j) + (ub(j)-lb(j)) * rand(sum(still_viol),1);
    end
    
    % Final bounds check
    offspring = max(min(offspring, ub), lb);
end
% MATLAB Code
function [offspring] = updateFunc1637(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Create elite pool considering both fitness and constraints
    weighted_fits = popfits + 1e6 * max(0, cons);
    [~, sorted_idx] = sort(weighted_fits);
    elite_size = max(2, ceil(NP*0.2));
    elite_pool = popdecs(sorted_idx(1:elite_size), :);
    
    % 2. Compute adaptive parameters
    [~, rank_order] = sort(weighted_fits);
    ranks = zeros(NP, 1);
    ranks(rank_order) = (1:NP)';
    
    c_abs = abs(cons);
    c_max = max(c_abs) + eps;
    
    % Adaptive F and CR parameters
    F = 0.5 + 0.3*(1 - ranks./NP) + 0.2*(1 - c_abs./c_max);
    CR = 0.9 * (1 - ranks./NP) .* (1 - c_abs./c_max);
    
    % 3. Generate offspring (vectorized operations)
    offspring = zeros(NP, D);
    for i = 1:NP
        % Select elite vector
        elite_idx = randi(elite_size);
        elite = elite_pool(elite_idx, :);
        
        % Select four distinct random vectors (excluding current)
        candidates = setdiff(1:NP, i);
        idx = candidates(randperm(length(candidates), 4));
        r1 = idx(1); r2 = idx(2); r3 = idx(3); r4 = idx(4);
        
        % Composite mutation with adaptive scaling
        mutation = elite + F(i)*(popdecs(r1,:)-popdecs(r2,:)) + ...
                  (1-F(i))*(popdecs(r3,:)-popdecs(r4,:));
        
        % Constraint-aware crossover with guaranteed change
        mask = rand(1,D) < CR(i);
        j_rand = randi(D);
        mask(j_rand) = true;
        offspring(i,:) = popdecs(i,:).*(~mask) + mutation.*mask;
    end
    
    % 4. Enhanced boundary handling
    % Reflection for moderate violations
    viol_low = offspring < lb;
    offspring(viol_low) = 2*lb(viol_low) - offspring(viol_low);
    
    viol_high = offspring > ub;
    offspring(viol_high) = 2*ub(viol_high) - offspring(viol_high);
    
    % Random reset for extreme violations
    still_violating = (offspring < lb) | (offspring > ub);
    if any(still_violating(:))
        offspring(still_violating) = lb(still_violating) + ...
            rand(sum(still_violating(:)),1).*(ub(still_violating)-lb(still_violating));
    end
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb), ub);
end
% MATLAB Code
function [offspring] = updateFunc1633(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Create elite pool considering both fitness and constraints
    weighted_fits = popfits + 1e6 * max(0, cons);
    [~, sorted_idx] = sort(weighted_fits);
    elite_size = max(2, ceil(NP*0.2));
    elite_pool = popdecs(sorted_idx(1:elite_size), :);
    
    % 2. Compute adaptive parameters
    [~, rank_order] = sort(popfits);
    ranks = zeros(NP, 1);
    ranks(rank_order) = (1:NP)';
    
    c_abs = abs(cons);
    c_max = max(c_abs) + eps;
    
    % Adaptive parameters (vectorized)
    F_base = 0.5;
    F_rank = 0.3 * (1 - ranks./NP);
    F_cons = 0.2 * (1 - c_abs./c_max);
    F = F_base + F_rank + F_cons;
    
    CR = 0.9 * (1 - ranks./NP) .* (1 - c_abs./c_max);
    
    % 3. Generate offspring (fully vectorized)
    offspring = zeros(NP, D);
    for i = 1:NP
        % Select elite and random vectors
        elite_idx = randi(elite_size);
        elite = elite_pool(elite_idx, :);
        
        candidates = setdiff(1:NP, i);
        idx = candidates(randperm(length(candidates), 4));
        r1 = idx(1); r2 = idx(2); r3 = idx(3); r4 = idx(4);
        
        % Composite mutation with adaptive weights
        mutation = elite + F(i)*(popdecs(r1,:)-popdecs(r2,:)) + ...
                  (1-F(i))*(popdecs(r3,:)-popdecs(r4,:));
        
        % Adaptive crossover with guaranteed change
        mask = rand(1,D) < CR(i);
        j_rand = randi(D);
        mask(j_rand) = true;
        offspring(i,:) = popdecs(i,:).*(~mask) + mutation.*mask;
    end
    
    % 4. Boundary handling with reflection and random reset
    % Lower bound violations
    viol_low = offspring < lb;
    offspring(viol_low) = 2*lb(viol_low) - offspring(viol_low);
    
    % Upper bound violations
    viol_high = offspring > ub;
    offspring(viol_high) = 2*ub(viol_high) - offspring(viol_high);
    
    % Final bounds enforcement with random reset if still out of bounds
    out_of_bounds = (offspring < lb) | (offspring > ub);
    if any(out_of_bounds(:))
        offspring(out_of_bounds) = lb(out_of_bounds) + rand(sum(out_of_bounds(:)),1).*(ub(out_of_bounds)-lb(out_of_bounds));
    end
end
% MATLAB Code
function [offspring] = updateFunc1650(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Constrained fitness ranking
    penalty = 1e6 * max(0, cons);
    weighted_fits = popfits + penalty;
    [~, sorted_idx] = sort(weighted_fits);
    ranks = zeros(NP, 1);
    ranks(sorted_idx) = 1:NP;
    weights = (NP - ranks + 1) / NP;
    
    % 2. Elite pool (top 30%)
    elite_size = max(2, ceil(NP*0.3));
    elite_pool = popdecs(sorted_idx(1:elite_size), :);
    x_best = popdecs(sorted_idx(1), :);
    
    % 3. Constraint normalization
    c_abs = abs(cons);
    c_max = max(c_abs) + eps;
    c_normalized = c_abs ./ c_max;
    
    % 4. Adaptive parameters
    F = 0.5 + 0.4 * weights .* (1 - c_normalized);
    CR = 0.9 * weights .* (1 - c_normalized);
    
    % 5. Generate offspring (vectorized)
    offspring = zeros(NP, D);
    for i = 1:NP
        % Select elite vectors
        elite_idx = randperm(elite_size, 2);
        e1 = elite_pool(elite_idx(1), :);
        e2 = elite_pool(elidex(2), :);
        
        % Select random distinct vectors
        candidates = setdiff(1:NP, i);
        idx = candidates(randperm(length(candidates), 4));
        r1 = idx(1); r2 = idx(2); r3 = idx(3); r4 = idx(4);
        
        % Weighted difference
        w_diff = weights(r3)/(weights(r3)+weights(r4)+eps) * (popdecs(r3,:)-popdecs(r4,:));
        
        % Composite mutation
        mutation = popdecs(i,:) + F(i) * (...
            (x_best - popdecs(i,:)) + ...
            (e1 - e2) + ...
            w_diff);
        
        % Constraint-aware crossover
        mask = rand(1,D) < CR(i);
        j_rand = randi(D);
        mask(j_rand) = true;
        offspring(i,:) = popdecs(i,:).*(~mask) + mutation.*mask;
    end
    
    % 6. Boundary handling with adaptive reflection
    viol_low = offspring < lb;
    viol_high = offspring > ub;
    offspring(viol_low) = lb(viol_low) + rand(sum(viol_low(:)),1).*(ub(viol_low)-lb(viol_low));
    offspring(viol_high) = ub(viol_high) - rand(sum(viol_high(:)),1).*(ub(viol_high)-lb(viol_high));
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb), ub);
end
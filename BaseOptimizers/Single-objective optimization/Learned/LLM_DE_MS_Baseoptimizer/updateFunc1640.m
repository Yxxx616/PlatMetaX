% MATLAB Code
function [offspring] = updateFunc1640(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Create elite and worst pools
    weighted_fits = popfits + 1e6 * max(0, cons);
    [~, sorted_idx] = sort(weighted_fits);
    elite_size = max(2, ceil(NP*0.2));
    worst_size = max(2, ceil(NP*0.2));
    elite_pool = popdecs(sorted_idx(1:elite_size), :);
    worst_pool = popdecs(sorted_idx(end-worst_size+1:end), :);
    x_best = popdecs(sorted_idx(1), :);
    x_worst = popdecs(sorted_idx(end), :);
    
    % 2. Compute adaptive parameters
    [~, rank_order] = sort(weighted_fits);
    ranks = zeros(NP, 1);
    ranks(rank_order) = (1:NP)';
    
    c_abs = abs(cons);
    c_max = max(c_abs) + eps;
    f_max = max(popfits);
    f_min = min(popfits);
    f_range = f_max - f_min + eps;
    
    % Adaptive parameters
    F1 = 0.5 + 0.3*(1 - ranks./NP);
    F2 = 0.2 * (f_max - popfits)./f_range;
    F3 = 0.3 * randn(NP, 1);
    
    % 3. Generate offspring (vectorized operations)
    offspring = zeros(NP, D);
    for i = 1:NP
        % Select elite vector
        elite_idx = randi(elite_size);
        elite = elite_pool(elite_idx, :);
        
        % Select random vectors
        candidates = setdiff(1:NP, i);
        idx = candidates(randperm(length(candidates), 4));
        r1 = idx(1); r2 = idx(2); r3 = idx(3); r4 = idx(4);
        
        % Composite mutation
        mutation = elite + F1(i)*(popdecs(r1,:)-popdecs(r2,:)) + ...
                  F2(i)*(x_best - x_worst).*(1 - c_abs(i)/c_max) + ...
                  F3(i)*(popdecs(r3,:)-popdecs(r4,:));
        
        % Adaptive crossover
        CR = 0.9 * (1 - ranks(i)/NP) * (1 - c_abs(i)/c_max);
        mask = rand(1,D) < CR;
        j_rand = randi(D);
        mask(j_rand) = true;
        offspring(i,:) = popdecs(i,:).*(~mask) + mutation.*mask;
    end
    
    % 4. Boundary handling with reflection and random reset
    viol_low = offspring < lb;
    viol_high = offspring > ub;
    
    % Reflection for moderate violations
    offspring(viol_low) = 2*lb(viol_low) - offspring(viol_low);
    offspring(viol_high) = 2*ub(viol_high) - offspring(viol_high);
    
    % Random reset for remaining violations
    still_violating = (offspring < lb) | (offspring > ub);
    if any(still_violating(:))
        num_viol = sum(still_violating(:));
        offspring(still_violating) = lb(still_violating) + ...
            rand(num_viol,1).*(ub(still_violating)-lb(still_violating));
    end
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb), ub);
end
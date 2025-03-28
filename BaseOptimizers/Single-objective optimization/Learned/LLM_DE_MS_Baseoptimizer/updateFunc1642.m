% MATLAB Code
function [offspring] = updateFunc1642(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Create elite and worst pools with constraint handling
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
    
    % Enhanced adaptive parameters
    F1 = 0.7 + 0.2*(1 - ranks./NP);
    F2 = 0.5 * (f_max - popfits)./f_range;
    F3 = 0.3 * randn(NP, 1);
    fit_ratio = (popfits - f_min)./f_range;
    
    % 3. Generate offspring (fully vectorized)
    offspring = zeros(NP, D);
    for i = 1:NP
        % Select elite vector
        elite_idx = randi(elite_size);
        elite = elite_pool(elite_idx, :);
        
        % Select random distinct vectors
        candidates = setdiff(1:NP, i);
        idx = candidates(randperm(length(candidates), 4));
        r1 = idx(1); r2 = idx(2); r3 = idx(3); r4 = idx(4);
        
        % Composite mutation with enhanced exploration
        mutation = elite + F1(i)*(popdecs(r1,:)-popdecs(r2,:)) + ...
                  F2(i)*(x_best - x_worst).*(1 - c_abs(i)/c_max) + ...
                  F3(i)*(popdecs(r3,:)-popdecs(r4,:)).*fit_ratio(i);
        
        % Adaptive crossover with constraint awareness
        CR = 0.9 * (1 - ranks(i)/NP)^1.5 * (1 - c_abs(i)/c_max);
        mask = rand(1,D) < CR;
        j_rand = randi(D);
        mask(j_rand) = true;
        offspring(i,:) = popdecs(i,:).*(~mask) + mutation.*mask;
    end
    
    % 4. Advanced boundary handling with reflection
    viol_low = offspring < lb;
    viol_high = offspring > ub;
    
    % Reflection with adaptive scaling
    offspring(viol_low) = lb(viol_low) + abs(offspring(viol_low) - lb(viol_low)).*rand(sum(viol_low(:)),1);
    offspring(viol_high) = ub(viol_high) - abs(offspring(viol_high) - ub(viol_high)).*rand(sum(viol_high(:)),1);
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb), ub);
end
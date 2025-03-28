% MATLAB Code
function [offspring] = updateFunc1649(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Sort population by constrained fitness
    weighted_fits = popfits + 1e6 * max(0, cons);
    [~, sorted_idx] = sort(weighted_fits);
    ranks = zeros(NP, 1);
    ranks(sorted_idx) = 1:NP;
    
    % 2. Create elite pool (top 20%)
    elite_size = max(2, ceil(NP*0.2));
    elite_pool = popdecs(sorted_idx(1:elite_size), :);
    x_best = popdecs(sorted_idx(1), :);
    
    % 3. Compute constraint awareness factors
    c_abs = abs(cons);
    c_max = max(c_abs) + eps;
    alpha = 1 - c_abs./c_max;
    
    % 4. Compute fitness weighting factors
    f_min = min(popfits);
    f_max = max(popfits);
    beta = (f_max - popfits) ./ (f_max - f_min + eps);
    
    % 5. Generate offspring (fully vectorized)
    offspring = zeros(NP, D);
    for i = 1:NP
        % Select elite vector
        elite_idx = randi(elite_size);
        elite = elite_pool(elite_idx, :);
        
        % Select random distinct vectors
        candidates = setdiff(1:NP, i);
        idx = candidates(randperm(length(candidates), 3));
        r1 = idx(1); r2 = idx(2); r3 = idx(3);
        
        % Adaptive scaling factors
        F1 = 0.9 - 0.5*(ranks(i)/NP)^0.7;
        F2 = 0.5*(1 - ranks(i)/NP)^0.5;
        F3 = 0.3 + 0.2*rand();
        
        % Composite mutation
        mutation = popdecs(i,:) + ...
                 F1*(elite - popdecs(i,:)) + ...
                 F2*alpha(i)*(popdecs(r1,:) - popdecs(r2,:)) + ...
                 F3*beta(i)*(x_best - popdecs(r3,:));
        
        % Adaptive crossover
        CR = 0.85*(1 - ranks(i)/NP)^0.4 * alpha(i)^0.6;
        mask = rand(1,D) < CR;
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
% MATLAB Code
function [offspring] = updateFunc1411(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    
    % Normalize constraints and fitness
    cv_pos = max(0, cons);
    cv_min = min(cv_pos);
    cv_max = max(cv_pos) + eps;
    cv_norm = (cv_pos - cv_min) / (cv_max - cv_min);
    
    f_min = min(popfits);
    f_max = max(popfits);
    f_range = f_max - f_min + eps;
    f_norm = (popfits - f_min) / f_range;
    
    % Find elite individual (considering both fitness and constraints)
    penalized_fits = popfits + 1e6 * cv_pos;
    [~, elite_idx] = min(penalized_fits);
    x_elite = popdecs(elite_idx, :);
    
    % Compute feasible center
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        x_feas = mean(popdecs(feasible_mask, :), 1);
    else
        [~, min_cv_idx] = min(cons);
        x_feas = popdecs(min_cv_idx, :);
    end
    
    % Generate random indices for diversity component
    rand_idx = zeros(NP, 2);
    for i = 1:NP
        candidates = setdiff(1:NP, [i, elite_idx]);
        if length(candidates) >= 2
            rand_idx(i,:) = candidates(randperm(length(candidates), 2));
        else
            rand_idx(i,:) = [1, 2]; % fallback
        end
    end
    
    % Adaptive scaling factors using tanh for smooth transition
    F = 0.5 * (1 + tanh(1 - cv_norm));
    
    % Compute mutation components
    diff_feas = bsxfun(@minus, x_feas, popdecs);
    diff_elite = bsxfun(@minus, x_elite, popdecs);
    diff_div = popdecs(rand_idx(:,1),:) - popdecs(rand_idx(:,2),:);
    
    % Weighted mutation
    w1 = (1 - cv_norm)/3;
    w2 = (1 - f_norm)/3;
    w3 = 1/3;
    mutants = popdecs + F .* (w1.*diff_feas + w2.*diff_elite + w3.*diff_div);
    
    % Boundary handling with reflection
    for j = 1:D
        below = mutants(:,j) < lb(j);
        above = mutants(:,j) > ub(j);
        mutants(below,j) = 2*lb(j) - mutants(below,j);
        mutants(above,j) = 2*ub(j) - mutants(above,j);
        
        % Additional check for extreme cases
        still_below = mutants(:,j) < lb(j);
        still_above = mutants(:,j) > ub(j);
        mutants(still_below,j) = lb(j) + rand(sum(still_below),1).*(ub(j)-lb(j))*0.1;
        mutants(still_above,j) = ub(j) - rand(sum(still_above),1).*(ub(j)-lb(j))*0.1;
    end
    
    % Adaptive crossover
    CR = 0.9 - 0.4 * cv_norm;
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Special handling for highly constrained individuals
    extreme_cv = cv_norm > 0.9;
    if any(extreme_cv)
        n_extreme = sum(extreme_cv);
        offspring(extreme_cv,:) = bsxfun(@plus, x_feas, ...
            bsxfun(@times, (ub - lb), rand(n_extreme,D)-0.5) .* 0.1);
    end
end
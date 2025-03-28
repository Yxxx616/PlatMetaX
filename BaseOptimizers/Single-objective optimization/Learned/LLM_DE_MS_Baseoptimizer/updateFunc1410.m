% MATLAB Code
function [offspring] = updateFunc1410(popdecs, popfits, cons)
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
    
    % Find reference points
    penalized_fits = popfits + 1e6 * cv_pos;
    [~, best_idx] = min(penalized_fits);
    x_best = popdecs(best_idx, :);
    
    % Feasible center
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        x_feas = mean(popdecs(feasible_mask, :), 1);
    else
        [~, min_cv_idx] = min(cons);
        x_feas = popdecs(min_cv_idx, :);
    end
    
    % Generate random indices for diversity
    rand_idx = zeros(NP, 2);
    for i = 1:NP
        candidates = setdiff(1:NP, [i, best_idx]);
        if length(candidates) >= 2
            rand_idx(i,:) = candidates(randperm(length(candidates), 2));
        else
            rand_idx(i,:) = [1, 2]; % fallback
        end
    end
    
    % Adaptive scaling factors
    F = 0.5 ./ (1 + cv_norm);
    
    % Mutation components
    diff_feas = bsxfun(@minus, x_feas, popdecs);
    diff_best = bsxfun(@minus, x_best, popdecs);
    diff_rand = popdecs(rand_idx(:,1),:) - popdecs(rand_idx(:,2),:);
    
    % Weighted mutation (equal weights)
    mutants = popdecs + F .* ((1-cv_norm)/3 .* diff_feas + ...
                             (1-f_norm)/3 .* diff_best + ...
                             1/3 .* diff_rand);
    
    % Boundary handling
    for j = 1:D
        below = mutants(:,j) < lb(j);
        above = mutants(:,j) > ub(j);
        mutants(below,j) = lb(j) + rand(sum(below),1) .* (ub(j)-lb(j))*0.1;
        mutants(above,j) = ub(j) - rand(sum(above),1) .* (ub(j)-lb(j))*0.1;
    end
    
    % Adaptive crossover
    CR = 0.9 - 0.5 * cv_norm;
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Special handling for extreme cases
    extreme_cv = cv_norm > 0.95;
    if any(extreme_cv)
        n_extreme = sum(extreme_cv);
        offspring(extreme_cv,:) = bsxfun(@plus, x_feas, ...
            bsxfun(@times, (ub - lb), rand(n_extreme,D)-0.5) .* 0.2);
    end
end
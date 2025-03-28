% MATLAB Code
function [offspring] = updateFunc1384(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    eps = 1e-12;
    
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
    [~, best_idx] = min(popfits + 1e6 * cv_pos);
    x_best = popdecs(best_idx, :);
    
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        x_feas = mean(popdecs(feasible_mask, :), 1);
    else
        [~, min_cv_idx] = min(cons);
        x_feas = popdecs(min_cv_idx, :);
    end
    
    % Generate random indices ensuring they're distinct
    rand_idx1 = randi(NP, NP, 1);
    rand_idx2 = randi(NP, NP, 1);
    for i = 1:NP
        while rand_idx1(i) == i || rand_idx1(i) == best_idx
            rand_idx1(i) = randi(NP);
        end
        while rand_idx2(i) == i || rand_idx2(i) == best_idx || rand_idx2(i) == rand_idx1(i)
            rand_idx2(i) = randi(NP);
        end
    end
    
    % Adaptive weights
    w_best = 0.6 * (1 - cv_norm) .* (1 - f_norm);
    w_feas = 0.4 * cv_norm .* (1 - f_norm);
    w_div = 0.2 * f_norm + 0.1 * cv_norm;
    
    % Mutation with adaptive components
    diff_best = x_best - popdecs;
    diff_feas = x_feas - popdecs;
    diff_rand = popdecs(rand_idx1,:) - popdecs(rand_idx2,:);
    
    mutants = popdecs + bsxfun(@times, w_best, diff_best) + ...
              bsxfun(@times, w_feas, diff_feas) + ...
              bsxfun(@times, w_div, diff_rand);
    
    % Boundary handling with adaptive reflection
    reflect_scale = 0.5 + 0.5 * rand(NP, 1);
    for j = 1:D
        below = mutants(:,j) < lb(j);
        above = mutants(:,j) > ub(j);
        
        mutants(below,j) = lb(j) + reflect_scale(below) .* (lb(j) - mutants(below,j));
        mutants(above,j) = ub(j) - reflect_scale(above) .* (mutants(above,j) - ub(j));
        
        % Final clamping
        mutants(below,j) = min(max(mutants(below,j), lb(j)), ub(j));
        mutants(above,j) = max(min(mutants(above,j), ub(j)), lb(j));
    end
    
    % Adaptive crossover with dynamic CR
    base_CR = 0.9;
    CR = base_CR * (1 - cv_norm) + 0.1;
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Final feasibility enforcement with diversity preservation
    out_of_bounds = any(offspring < lb | offspring > ub, 2);
    if any(out_of_bounds)
        feasible_pop = popdecs(feasible_mask,:);
        if isempty(feasible_pop)
            feasible_pop = popdecs;
        end
        centroid = mean(feasible_pop, 1);
        offspring(out_of_bounds,:) = bsxfun(@plus, centroid, ...
                                   0.1 * bsxfun(@times, (ub - lb), rand(sum(out_of_bounds),D)-0.5));
    end
end
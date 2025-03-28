% MATLAB Code
function [offspring] = updateFunc1387(popdecs, popfits, cons)
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
    
    % Adaptive weights
    w_explore = 0.4 * (1 - f_norm) .* cv_norm;
    w_exploit = 0.6 * (1 - cv_norm) .* (1 - f_norm);
    
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
    
    % Generate random indices
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
    
    % Mutation with adaptive components
    diff_best = x_best - popdecs;
    diff_feas = x_feas - popdecs;
    diff_rand = popdecs(rand_idx1,:) - popdecs(rand_idx2,:);
    
    mutants = popdecs + bsxfun(@times, w_exploit, diff_best) + ...
              bsxfun(@times, w_explore, diff_feas) + ...
              0.2 * diff_rand;
    
    % Adaptive boundary handling
    reflect_scale = 0.3 + 0.4 * rand(NP, 1);
    for j = 1:D
        below = mutants(:,j) < lb(j);
        above = mutants(:,j) > ub(j);
        
        mutants(below,j) = lb(j) + reflect_scale(below) .* (lb(j) - mutants(below,j));
        mutants(above,j) = ub(j) - reflect_scale(above) .* (mutants(above,j) - ub(j));
        
        % Final clamping
        mutants(:,j) = min(max(mutants(:,j), lb(j)), ub(j));
    end
    
    % Dynamic crossover
    base_CR = 0.85;
    CR = base_CR * (1 - cv_norm) + 0.15;
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Diversity preservation for out-of-bounds solutions
    out_of_bounds = any(offspring < lb | offspring > ub, 2);
    if any(out_of_bounds)
        if any(feasible_mask)
            centroid = x_feas;
        else
            centroid = mean(popdecs, 1);
        end
        offspring(out_of_bounds,:) = bsxfun(@plus, centroid, ...
                                   0.1 * bsxfun(@times, (ub - lb), rand(sum(out_of_bounds),D)-0.5));
    end
end
% MATLAB Code
function [offspring] = updateFunc1417(popdecs, popfits, cons)
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
    
    % Compute adaptive weights
    w_feas = 0.6 * (1 - cv_norm);
    w_elite = 0.3 * (1 - f_norm);
    w_div = 0.1 * sqrt(cv_norm .* f_norm);
    
    % Select elite individual (considering both fitness and constraints)
    [~, elite_idx] = min(popfits + 1e6*cv_pos);
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
    
    % Mutation components
    diff_feas = bsxfun(@minus, x_feas, popdecs);
    diff_elite = bsxfun(@minus, x_elite, popdecs);
    diff_div = popdecs(rand_idx(:,1),:) - popdecs(rand_idx(:,2),:);
    
    % Combined mutation with adaptive weights
    mutants = popdecs + w_feas.*diff_feas + w_elite.*diff_elite + w_div.*diff_div;
    
    % Boundary handling with reflection
    for j = 1:D
        below = mutants(:,j) < lb(j);
        above = mutants(:,j) > ub(j);
        mutants(below,j) = 2*lb(j) - mutants(below,j);
        mutants(above,j) = 2*ub(j) - mutants(above,j);
        
        % Random reset if still outside bounds
        out_of_bounds = (mutants(:,j) < lb(j)) | (mutants(:,j) > ub(j));
        mutants(out_of_bounds,j) = lb(j) + rand(sum(out_of_bounds),1).*(ub(j)-lb(j));
    end
    
    % Dynamic crossover rate based on normalized constraints
    CR = 0.9 - 0.5 * cv_norm;
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Enhanced exploration for poor solutions
    poor_thresh = 0.8;
    poor_mask = (cv_norm > poor_thresh) | (f_norm > poor_thresh);
    if any(poor_mask)
        n_poor = sum(poor_mask);
        offspring(poor_mask,:) = bsxfun(@plus, x_feas, ...
            bsxfun(@times, (ub - lb), rand(n_poor,D)-0.5) * 0.1);
    end
end
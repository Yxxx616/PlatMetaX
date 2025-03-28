% MATLAB Code
function [offspring] = updateFunc1739(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Process constraints
    pos_cons = max(0, cons);
    feasible_mask = pos_cons == 0;
    
    % Select base vector (best feasible or least infeasible)
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
        feas_idx = find(feasible_mask);
        base_vec = popdecs(feas_idx(best_idx),:);
    else
        [~, best_idx] = min(pos_cons);
        base_vec = popdecs(best_idx,:);
    end
    
    % Calculate population statistics
    feas_pop = popdecs(feasible_mask,:);
    infeas_pop = popdecs(~feasible_mask,:);
    
    feas_mean = mean(feas_pop, 1);
    if isempty(feas_mean)
        feas_mean = mean(popdecs, 1);
    end
    
    infeas_mean = mean(infeas_pop, 1);
    if isempty(infeas_mean)
        infeas_mean = zeros(1, D);
    end
    
    % Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fits = (popfits - f_min) / (f_max - f_min + eps);
    
    c_min = min(pos_cons);
    c_max = max(pos_cons);
    norm_cons = (pos_cons - c_min) / (c_max - c_min + eps);
    
    % Generate unique random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    mask = r1 == r2;
    while any(mask)
        r2(mask) = randi(NP, sum(mask), 1);
        mask = r1 == r2;
    end
    
    % Adaptive parameters with improved weighting
    F1 = 0.8 * (1 - norm_fits);
    F2 = 0.6 * (1 - norm_cons);
    F3 = 0.4 * norm_cons;
    CR = 0.9 - 0.5 * norm_cons;
    
    % Vectorized mutation with improved exploration
    base_rep = repmat(base_vec, NP, 1);
    feas_mean_rep = repmat(feas_mean, NP, 1);
    infeas_mean_rep = repmat(infeas_mean, NP, 1);
    
    mutation = base_rep + ...
               F1 .* (feas_mean_rep - popdecs) + ...
               F2 .* (popdecs(r1,:) - popdecs(r2,:)) + ...
               F3 .* (feas_mean_rep - infeas_mean_rep);
    
    % Binomial crossover with jitter
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) <= CR(:, ones(1,D)) | bsxfun(@eq, (1:D), j_rand);
    offspring = popdecs;
    offspring(mask) = mutation(mask);
    
    % Boundary handling with reflection
    out_of_bounds = offspring < lb | offspring > ub;
    offspring(out_of_bounds) = lb(out_of_bounds) + rand(sum(out_of_bounds(:)),1) .* ...
                              (ub(out_of_bounds) - lb(out_of_bounds));
    
    % Elite preservation for best solution
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
        feas_idx = find(feasible_mask);
        offspring(feas_idx(best_idx),:) = popdecs(feas_idx(best_idx),:);
    else
        [~, best_idx] = min(pos_cons);
        offspring(best_idx,:) = popdecs(best_idx,:);
    end
end
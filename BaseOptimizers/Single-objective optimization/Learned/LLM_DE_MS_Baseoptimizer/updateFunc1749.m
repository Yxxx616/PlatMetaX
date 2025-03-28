% MATLAB Code
function [offspring] = updateFunc1749(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Process constraints
    pos_cons = max(0, cons);
    feasible_mask = pos_cons == 0;
    feasible_ratio = sum(feasible_mask)/NP;
    
    % Select elite vector (best feasible or least infeasible)
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        feas_idx = find(feasible_mask);
        elite_vec = popdecs(feas_idx(elite_idx),:);
    else
        [~, elite_idx] = min(pos_cons);
        elite_vec = popdecs(elite_idx,:);
    end
    
    % Select best and worst by constraints
    [~, best_cons_idx] = min(pos_cons);
    [~, worst_cons_idx] = max(pos_cons);
    
    % Calculate weighted mean vectors
    if any(feasible_mask)
        feas_mean = mean(popdecs(feasible_mask,:));
    else
        feas_mean = mean(popdecs);
    end
    
    if any(~feasible_mask)
        weights = 1./(pos_cons(~feasible_mask)+eps);
        weights = weights/sum(weights);
        infeas_mean = sum(popdecs(~feasible_mask,:) .* weights(:, ones(1,D)), 1);
    else
        infeas_mean = zeros(1, D);
    end
    
    % Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fits = (popfits - f_min) / (f_max - f_min + eps);
    
    c_max = max(pos_cons);
    norm_cons = pos_cons / (c_max + eps);
    
    % Generate random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    mask = r1 == r2;
    while any(mask)
        r2(mask) = randi(NP, sum(mask), 1);
        mask = r1 == r2;
    end
    
    % Adaptive scaling factors
    F1 = 0.6 * (1 - norm_fits);
    F2 = 0.7 * feasible_ratio;
    F3 = 0.5 * norm_cons;
    F4 = 0.4 * rand(NP, 1);
    
    % Vectorized mutation
    elite_rep = repmat(elite_vec, NP, 1);
    feas_mean_rep = repmat(feas_mean, NP, 1);
    infeas_mean_rep = repmat(infeas_mean, NP, 1);
    best_cons_rep = repmat(popdecs(best_cons_idx,:), NP, 1);
    worst_cons_rep = repmat(popdecs(worst_cons_idx,:), NP, 1);
    
    mutation = elite_rep + ...
               F1(:, ones(1,D)) .* (elite_rep - popdecs) + ...
               F2 * (feas_mean_rep - infeas_mean_rep) + ...
               F3(:, ones(1,D)) .* (best_cons_rep - worst_cons_rep) + ...
               F4(:, ones(1,D)) .* (popdecs(r1,:) - popdecs(r2,:));
    
    % Adaptive crossover with constraint-awareness
    CR = 0.9 - 0.4 * norm_cons;
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) <= CR(:, ones(1,D)) | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutation(mask);
    
    % Smart boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    lower_violation = offspring < lb_rep;
    upper_violation = offspring > ub_rep;
    
    offspring(lower_violation) = lb_rep(lower_violation) + ...
        rand(sum(sum(lower_violation)),1) .* ...
        (popdecs(lower_violation) - lb_rep(lower_violation));
    offspring(upper_violation) = ub_rep(upper_violation) - ...
        rand(sum(sum(upper_violation)),1) .* ...
        (ub_rep(upper_violation) - popdecs(upper_violation));
    
    % Preserve elite solution
    if any(feasible_mask)
        offspring(feas_idx(elite_idx),:) = popdecs(feas_idx(elite_idx),:);
    else
        offspring(elite_idx,:) = popdecs(elite_idx,:);
    end
end
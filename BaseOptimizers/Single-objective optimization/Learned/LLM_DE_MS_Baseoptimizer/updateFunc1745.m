% MATLAB Code
function [offspring] = updateFunc1745(popdecs, popfits, cons)
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
    
    % Calculate mean vectors
    if any(feasible_mask)
        feas_mean = mean(popdecs(feasible_mask,:));
    else
        feas_mean = zeros(1, D);
    end
    
    if any(~feasible_mask)
        weights = pos_cons(~feasible_mask)/sum(pos_cons(~feasible_mask));
        infeas_mean = sum(popdecs(~feasible_mask,:) .* weights(:, ones(1,D)), 1);
    else
        infeas_mean = zeros(1, D);
    end
    
    % Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fits = (popfits - f_min) / (f_max - f_min + eps);
    
    c_max = max(pos_cons);
    cons_ratio = pos_cons / (c_max + eps);
    
    % Generate random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    mask = r1 == r2;
    while any(mask)
        r2(mask) = randi(NP, sum(mask), 1);
        mask = r1 == r2;
    end
    
    % Adaptive scaling factors
    F1 = 0.5 * (1 - norm_fits);
    F2 = 0.8 * feasible_ratio;
    F3 = 0.6 * (1 - feasible_ratio);
    F4 = 0.5 * rand(NP, 1);
    F5 = 0.3 * cons_ratio;
    
    % Vectorized mutation
    elite_rep = repmat(elite_vec, NP, 1);
    feas_mean_rep = repmat(feas_mean, NP, 1);
    infeas_mean_rep = repmat(infeas_mean, NP, 1);
    best_cons_rep = repmat(popdecs(best_cons_idx,:), NP, 1);
    worst_cons_rep = repmat(popdecs(worst_cons_idx,:), NP, 1);
    
    mutation = elite_rep + ...
               F1(:, ones(1,D)) .* (elite_rep - popdecs) + ...
               F2 .* (feas_mean_rep - popdecs) + ...
               F3 .* (popdecs - infeas_mean_rep) + ...
               F4(:, ones(1,D)) .* (popdecs(r1,:) - popdecs(r2,:)) + ...
               F5(:, ones(1,D)) .* (best_cons_rep - worst_cons_rep);
    
    % Adaptive crossover
    CR = 0.9 - 0.5 * cons_ratio;
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) <= CR(:, ones(1,D)) | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutation(mask);
    
    % Boundary handling with bounce-back
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
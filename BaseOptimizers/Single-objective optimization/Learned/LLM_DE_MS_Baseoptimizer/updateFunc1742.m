% MATLAB Code
function [offspring] = updateFunc1742(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Process constraints
    pos_cons = max(0, cons);
    feasible_mask = pos_cons == 0;
    infeasible_mask = ~feasible_mask;
    
    % Select target vector
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
        feas_idx = find(feasible_mask);
        target_vec = popdecs(feas_idx(best_idx),:);
    else
        [~, best_idx] = min(pos_cons);
        target_vec = popdecs(best_idx,:);
    end
    
    % Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fits = (popfits - f_min) / (f_max - f_min + eps);
    
    c_max = max(pos_cons);
    norm_cons = pos_cons / (c_max + eps);
    
    % Calculate direction vectors
    if any(feasible_mask)
        d_f = mean(popdecs(feasible_mask,:) - target_vec, 1);
    else
        d_f = zeros(1, D);
    end
    
    if any(infeasible_mask)
        weights = pos_cons(infeasible_mask) / sum(pos_cons(infeasible_mask));
        d_c = sum((popdecs(infeasible_mask,:) - target_vec) .* weights(:, ones(1,D)), 1);
    else
        d_c = zeros(1, D);
    end
    
    % Generate random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    mask = r1 == r2;
    while any(mask)
        r2(mask) = randi(NP, sum(mask), 1);
        mask = r1 == r2;
    end
    
    % Adaptive scaling factors
    F_f = 0.8 * (1 - norm_cons);
    F_c = 0.6 * norm_cons;
    F_r = 0.5 * (1 - norm_fits);
    
    % Vectorized mutation
    target_rep = repmat(target_vec, NP, 1);
    d_f_rep = repmat(d_f, NP, 1);
    d_c_rep = repmat(d_c, NP, 1);
    
    mutation = target_rep + ...
               F_f(:, ones(1,D)) .* d_f_rep + ...
               F_c(:, ones(1,D)) .* d_c_rep + ...
               F_r(:, ones(1,D)) .* (popdecs(r1,:) - popdecs(r2,:));
    
    % Adaptive crossover
    CR = 0.9 - 0.5 * norm_cons;
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) <= CR(:, ones(1,D)) | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutation(mask);
    
    % Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    lower_violation = offspring < lb_rep;
    upper_violation = offspring > ub_rep;
    
    offspring(lower_violation) = 2*lb_rep(lower_violation) - offspring(lower_violation);
    offspring(upper_violation) = 2*ub_rep(upper_violation) - offspring(upper_violation);
    
    % Final clipping
    offspring = max(min(offspring, ub_rep), lb_rep);
    
    % Preserve best solution
    if any(feasible_mask)
        offspring(feas_idx(best_idx),:) = popdecs(feas_idx(best_idx),:);
    else
        offspring(best_idx,:) = popdecs(best_idx,:);
    end
end
% MATLAB Code
function [offspring] = updateFunc1807(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Process constraints
    pos_cons = max(0, cons);
    feasible_mask = pos_cons == 0;
    feasible_ratio = sum(feasible_mask)/NP;
    
    % Normalize constraints and fitness
    c_max = max(pos_cons);
    norm_cons = pos_cons / (c_max + eps);
    norm_fits = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    
    % Elite selection (best feasible or least infeasible)
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        feas_idx = find(feasible_mask);
        elite_vec = popdecs(feas_idx(elite_idx),:);
    else
        [~, elite_idx] = min(pos_cons);
        elite_vec = popdecs(elite_idx,:);
    end
    
    % Best fitness individual
    [~, best_fit_idx] = min(popfits);
    best_fit_vec = popdecs(best_fit_idx,:);
    
    % Calculate centroids
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
    
    % Constraint extremes
    [~, best_cons_idx] = min(pos_cons);
    [~, worst_cons_idx] = max(pos_cons);
    
    % Generate distinct random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    mask = r1 == r2;
    while any(mask)
        r2(mask) = randi(NP, sum(mask), 1);
        mask = r1 == r2;
    end
    
    % Adaptive scaling factors
    F1 = 0.8 + 0.1 * rand(NP, 1);  % Strong elite guidance
    F2 = 0.6 * feasible_ratio * ones(NP, 1);
    F3 = 0.5 * (1 - norm_cons).^2;
    F4 = 0.4 + 0.2 * rand(NP, 1);  % Balanced exploration
    F5 = 0.3 * (1 - norm_fits) .* rand(NP, 1);
    F6 = 0.2 * (1 - feasible_ratio) .* rand(NP, 1);
    
    % Vectorized mutation
    elite_rep = repmat(elite_vec, NP, 1);
    feas_mean_rep = repmat(feas_mean, NP, 1);
    infeas_mean_rep = repmat(infeas_mean, NP, 1);
    best_cons_rep = repmat(popdecs(best_cons_idx,:), NP, 1);
    worst_cons_rep = repmat(popdecs(worst_cons_idx,:), NP, 1);
    best_fit_rep = repmat(best_fit_vec, NP, 1);
    
    mutation = elite_rep + ...
               F1(:, ones(1,D)) .* (elite_rep - popdecs) + ...
               F2(:, ones(1,D)) .* (feas_mean_rep - infeas_mean_rep) + ...
               F3(:, ones(1,D)) .* (best_cons_rep - worst_cons_rep) .* (1 + norm_cons(:, ones(1,D))) + ...
               F4(:, ones(1,D)) .* (popdecs(r1,:) - popdecs(r2,:)) + ...
               F5(:, ones(1,D)) .* (best_fit_rep - popdecs) .* (1 - norm_fits(:, ones(1,D))) + ...
               F6(:, ones(1,D)) .* randn(NP, D) .* (elite_rep - feas_mean_rep);
    
    % Adaptive crossover
    CR = 0.9 - 0.4 * norm_cons;
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
    
    % Ensure within bounds
    offspring = min(max(offspring, lb_rep), ub_rep);
    
    % Elite preservation (top 10% solutions)
    preserve_num = max(1, floor(NP*0.1));
    if any(feasible_mask)
        [~, sort_idx] = sort(popfits(feasible_mask));
        feas_idx = find(feasible_mask);
        elite_indices = feas_idx(sort_idx(1:preserve_num));
    else
        [~, sort_idx] = sort(pos_cons);
        elite_indices = sort_idx(1:preserve_num);
    end
    offspring(elite_indices,:) = popdecs(elite_indices,:);
    
    % Adaptive perturbation
    if feasible_ratio > 0.6
        perturb_num = max(1, floor(NP*0.1));
        perturb_idx = randperm(NP, perturb_num);
        offspring(perturb_idx,:) = offspring(perturb_idx,:) + ...
            randn(perturb_num,D) .* (ub-lb)*0.003;
    end
end
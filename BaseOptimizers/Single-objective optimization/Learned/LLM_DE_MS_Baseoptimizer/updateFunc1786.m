% MATLAB Code
function [offspring] = updateFunc1786(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Process constraints
    pos_cons = max(0, cons);
    feasible_mask = pos_cons == 0;
    feasible_ratio = sum(feasible_mask)/NP;
    
    % Enhanced elite selection with memory
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        feas_idx = find(feasible_mask);
        elite_vec = popdecs(feas_idx(elite_idx),:);
    else
        [~, elite_idx] = min(pos_cons);
        elite_vec = popdecs(elite_idx,:);
    end
    
    % Improved centroid calculation
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
    
    % Constraint-based selection
    [~, best_cons_idx] = min(pos_cons);
    [~, worst_cons_idx] = max(pos_cons);
    
    % Enhanced normalization
    f_min = min(popfits);
    f_max = max(popfits);
    norm_fits = (popfits - f_min) / (f_max - f_min + eps);
    
    c_max = max(pos_cons);
    norm_cons = pos_cons / (c_max + eps);
    
    % Generate distinct random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    mask = r1 == r2;
    while any(mask)
        r2(mask) = randi(NP, sum(mask), 1);
        mask = r1 == r2;
    end
    
    % Optimized scaling factors
    F1 = 0.8 + 0.1 * (1 - norm_fits);
    F2 = 0.6 * feasible_ratio * ones(NP, 1);
    F3 = 0.4 * (1 - norm_cons);
    F4 = 0.2 * rand(NP, 1);
    F5 = 0.1 * rand(NP, 1);
    
    % Vectorized mutation with improved components
    elite_rep = repmat(elite_vec, NP, 1);
    feas_mean_rep = repmat(feas_mean, NP, 1);
    infeas_mean_rep = repmat(infeas_mean, NP, 1);
    best_cons_rep = repmat(popdecs(best_cons_idx,:), NP, 1);
    worst_cons_rep = repmat(popdecs(worst_cons_idx,:), NP, 1);
    
    mutation = elite_rep + ...
               F1(:, ones(1,D)) .* (elite_rep - popdecs) + ...
               F2(:, ones(1,D)) .* (feas_mean_rep - infeas_mean_rep) + ...
               F3(:, ones(1,D)) .* (best_cons_rep - worst_cons_rep) + ...
               F4(:, ones(1,D)) .* (popdecs(r1,:) - popdecs(r2,:)) + ...
               F5(:, ones(1,D)) .* randn(NP, D);
    
    % Dynamic crossover rate
    CR = 0.9 - 0.5 * norm_cons;
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) <= CR(:, ones(1,D)) | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutation(mask);
    
    % Smart boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    lower_violation = offspring < lb_rep;
    upper_violation = offspring > ub_rep;
    
    offspring(lower_violation) = lb_rep(lower_violation) + rand(sum(lower_violation(:)),1) .* ...
                                (elite_vec(mod(find(lower_violation')-1,D)+1) - lb_rep(lower_violation));
    offspring(upper_violation) = ub_rep(upper_violation) - rand(sum(upper_violation(:)),1) .* ...
                                (ub_rep(upper_violation) - elite_vec(mod(find(upper_violation')-1,D)+1));
    
    % Elite preservation with dynamic size
    preserve_num = max(1, min(5, floor(NP*0.1)));
    if any(feasible_mask)
        [~, sort_idx] = sort(popfits(feasible_mask));
        feas_idx = find(feasible_mask);
        offspring(feas_idx(sort_idx(1:preserve_num)),:) = popdecs(feas_idx(sort_idx(1:preserve_num)),:);
    else
        [~, sort_idx] = sort(pos_cons);
        offspring(sort_idx(1:preserve_num),:) = popdecs(sort_idx(1:preserve_num),:);
    end
    
    % Adaptive diversity maintenance
    if feasible_ratio < 0.3
        perturb_idx = randperm(NP, max(1, floor(NP*0.2)));
        offspring(perturb_idx,:) = offspring(perturb_idx,:) + ...
                                 randn(length(perturb_idx),D) .* (ub-lb)*0.1;
    end
end
% MATLAB Code
function [offspring] = updateFunc145(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Enhanced normalization with protection
    min_fit = min(popfits);
    max_fit = max(popfits);
    min_con = min(cons);
    max_con = max(cons);
    
    norm_fits = (popfits - min_fit) / (max_fit - min_fit + eps);
    norm_cons = (cons - min_con) / (max_con - min_con + eps);
    
    % Population partitioning
    [~, sorted_idx] = sort(norm_fits);
    elite_size = max(2, floor(0.2 * NP));
    elite_idx = sorted_idx(1:elite_size);
    x_elite = mean(popdecs(elite_idx,:), 1);
    x_mean = mean(popdecs, 1);
    
    % Best and worst individuals
    [~, best_idx] = min(norm_fits);
    [~, worst_idx] = max(norm_fits);
    x_best = popdecs(best_idx,:);
    x_worst = popdecs(worst_idx,:);
    
    % Random indices matrix (vectorized)
    rand_idx = randi(NP, NP, 3);
    r1 = rand_idx(:,1); 
    r2 = rand_idx(:,2); 
    r3 = rand_idx(:,3);
    
    % Adaptive parameters
    F1 = 0.8 * (1 - norm_fits);
    F2 = 0.5 + 0.3 * rand(NP, 1);
    F3 = 0.3 * (1 - norm_cons);
    CR = 0.9 - 0.5 * norm_cons;
    
    % Mutation components
    v_elite = x_elite + F1 .* (x_elite - x_mean);
    v_cons = F2 .* (popdecs(r1,:) - popdecs(r2,:)) .* (1 - norm_cons);
    v_div = F3 .* (x_best - x_worst) .* norm_fits;
    
    % Mutation strategy selection
    v = zeros(NP, D);
    low_fit = norm_fits < 0.3;
    mid_fit = (norm_fits >= 0.3) & (norm_fits <= 0.7);
    
    v(low_fit,:) = v_elite(low_fit,:) + v_cons(low_fit,:);
    v(mid_fit,:) = popdecs(mid_fit,:) + v_cons(mid_fit,:) + v_div(mid_fit,:);
    v(~low_fit & ~mid_fit,:) = popdecs(r3(~low_fit & ~mid_fit),:) + v_cons(~low_fit & ~mid_fit,:);
    
    % Crossover with adaptive CR
    CR_mat = repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR_mat;
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = v(mask);
    
    % Boundary handling with adaptive reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    reflect_prob = 0.6 + 0.3 * norm_fits;
    reflect_mask = rand(NP, D) < repmat(reflect_prob, 1, D);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    % Reflection for selected violations
    offspring(below_lb & reflect_mask) = 2*lb_rep(below_lb & reflect_mask) - offspring(below_lb & reflect_mask);
    offspring(above_ub & reflect_mask) = 2*ub_rep(above_ub & reflect_mask) - offspring(above_ub & reflect_mask);
    
    % Random reinitialization for remaining violations
    out_of_bounds = (offspring < lb_rep) | (offspring > ub_rep);
    offspring(out_of_bounds) = lb_rep(out_of_bounds) + rand(sum(out_of_bounds(:)),1) .* ...
                              (ub_rep(out_of_bounds) - lb_rep(out_of_bounds));
end
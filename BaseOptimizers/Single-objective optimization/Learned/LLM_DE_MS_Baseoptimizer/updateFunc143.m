% MATLAB Code
function [offspring] = updateFunc143(popdecs, popfits, cons)
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
    
    % Weighted score combining fitness and constraints
    scores = 0.7 * (1 - norm_fits) + 0.3 * (1 - norm_cons);
    
    % Improved population partitioning
    [~, sorted_idx] = sort(scores);
    elite_size = max(2, floor(0.2 * NP));
    mod_size = floor(0.6 * NP);
    elite_idx = sorted_idx(1:elite_size);
    mod_idx = sorted_idx(elite_size+1:elite_size+mod_size);
    poor_idx = sorted_idx(elite_size+mod_size+1:end);
    
    % Centroid calculations with weighted means
    x_elite = mean(popdecs(elite_idx,:), 1);
    x_mod = mean(popdecs(mod_idx,:), 1);
    x_poor = mean(popdecs(poor_idx,:), 1);
    
    % Random indices matrix (vectorized)
    rand_idx = randi(NP, NP, 3);
    r1 = rand_idx(:,1); 
    r2 = rand_idx(:,2); 
    r3 = rand_idx(:,3);
    
    % Adaptive parameters with improved scaling
    F1 = 0.6 * (1 + norm_fits);
    F2 = 0.3 + 0.4 * rand(NP, 1);
    F3 = 0.2 * (1 - norm_cons);
    CR = 0.9 - 0.5 * norm_cons;
    
    % Mutation vectors (fully vectorized)
    v_cons = F2 .* (x_mod - x_poor) .* (1 - norm_cons);
    v_div = F3 .* (popdecs(r1,:) - popdecs(r2,:)) .* norm_fits;
    
    % Mutation strategy selection
    v = zeros(NP, D);
    is_elite = ismember(1:NP, elite_idx)';
    is_mod = ismember(1:NP, mod_idx)';
    
    v(is_elite,:) = x_elite + F1(is_elite) .* (x_elite - x_poor) + v_cons(is_elite,:);
    v(is_mod,:) = popdecs(is_mod,:) + v_cons(is_mod,:) + v_div(is_mod,:);
    v(~is_elite & ~is_mod,:) = popdecs(r3(~is_elite & ~is_mod),:) + v_cons(~is_elite & ~is_mod,:);
    
    % Crossover with adaptive CR
    CR_mat = repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR_mat;
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = v(mask);
    
    % Enhanced boundary handling
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Dynamic reflection probability based on fitness
    reflect_prob = repmat(0.6 + 0.3 * norm_fits, 1, D);
    reflect_mask = rand(NP, D) < reflect_prob;
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    % Reflection for selected violations
    offspring(below_lb & reflect_mask) = 2*lb_rep(below_lb & reflect_mask) - offspring(below_lb & reflect_mask);
    offspring(above_ub & reflect_mask) = 2*ub_rep(above_ub & reflect_mask) - offspring(above_ub & reflect_mask);
    
    % Gaussian perturbation for remaining violations
    out_of_bounds = (offspring < lb_rep) | (offspring > ub_rep);
    gauss_pert = 0.1 * randn(NP, D) .* (ub_rep - lb_rep);
    offspring(out_of_bounds) = offspring(out_of_bounds) + gauss_pert(out_of_bounds);
    
    % Final clipping to ensure bounds
    offspring = max(min(offspring, ub_rep), lb_rep);
end
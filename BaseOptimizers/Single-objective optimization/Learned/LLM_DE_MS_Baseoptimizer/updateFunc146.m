% MATLAB Code
function [offspring] = updateFunc146(popdecs, popfits, cons)
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
    poor_idx = sorted_idx(end-elite_size+1:end);
    x_elite = mean(popdecs(elite_idx,:), 1);
    x_poor = mean(popdecs(poor_idx,:), 1);
    
    % Best individual
    [~, best_idx] = min(norm_fits);
    x_best = popdecs(best_idx,:);
    
    % Random indices matrix (vectorized)
    rand_idx = randi(NP, NP, 3);
    r1 = rand_idx(:,1); 
    r2 = rand_idx(:,2); 
    r3 = rand_idx(:,3);
    
    % Adaptive parameters
    F = 0.5 + 0.3 * norm_fits .* (1 - norm_cons);
    CR = 0.9 - 0.5 * norm_cons;
    
    % Mutation strategy selection
    v = zeros(NP, D);
    is_elite = false(NP,1);
    is_elite(elite_idx) = true;
    is_poor = false(NP,1);
    is_poor(poor_idx) = true;
    
    % Elite mutation
    v(is_elite,:) = x_elite + F(is_elite) .* (x_elite - x_poor) + ...
                   F(is_elite) .* (popdecs(r1(is_elite),:) - popdecs(r2(is_elite),:));
    
    % Middle population mutation
    mid_mask = ~is_elite & ~is_poor;
    v(mid_mask,:) = popdecs(mid_mask,:) + F(mid_mask) .* (x_best - popdecs(mid_mask,:)) + ...
                   F(mid_mask) .* (popdecs(r1(mid_mask),:) - popdecs(r2(mid_mask),:));
    
    % Poor population mutation
    v(is_poor,:) = popdecs(r1(is_poor),:) + F(is_poor) .* (popdecs(r2(is_poor),:) - popdecs(r3(is_poor),:));
    
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
    
    reflect_mask = rand(NP, D) < 0.7;
    
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
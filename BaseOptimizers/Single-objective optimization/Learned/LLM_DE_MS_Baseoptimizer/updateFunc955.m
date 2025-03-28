% MATLAB Code
function [offspring] = updateFunc955(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Base vector selection with feasibility consideration
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
        x_best = popdecs(feasible_mask,:);
        x_best = x_best(best_idx,:);
        x_centroid = mean(popdecs(feasible_mask,:), 1);
        alpha = 0.7; % stronger attraction to best
        x_base_feas = x_best + alpha*(x_centroid - x_best);
    else
        x_base_feas = mean(popdecs, 1);
    end
    
    % Least violated solution as secondary base
    [~, lv_idx] = min(cons);
    x_lv = popdecs(lv_idx,:);
    c_avg = mean(max(cons, 0));
    beta = min(cons(lv_idx)/(cons(lv_idx) + c_avg + eps);
    x_base_infeas = x_lv + beta*(x_best - x_lv);
    
    % Combine base vectors
    x_base = zeros(NP, D);
    for i = 1:NP
        if feasible_mask(i)
            x_base(i,:) = x_base_feas;
        else
            x_base(i,:) = x_base_infeas;
        end
    end
    
    % 2. Compute adaptive parameters
    f_min = min(popfits); f_max = max(popfits);
    c_min = min(cons); c_max = max(cons);
    
    F1 = 0.5 + 0.5*(popfits - f_min)./(f_max - f_min + eps);
    F2 = 0.5 - 0.5*(cons - c_min)./(c_max - c_min + eps);
    CR = 0.9 - 0.5*(cons./(c_max + eps));
    
    % 3. Generate random indices (vectorized)
    all_indices = 1:NP;
    r = zeros(NP, 4);
    for i = 1:NP
        candidates = setdiff(all_indices, i);
        r(i,:) = candidates(randperm(length(candidates), 4));
    end
    
    % 4. Enhanced directional mutation
    diff1 = popdecs(r(:,1),:) - popdecs(r(:,2),:);
    diff2 = popdecs(r(:,3),:) - popdecs(r(:,4),:);
    
    mutants = x_base + ...
              F1.*diff1 + ...
              F2.*diff2;
    
    % 5. Crossover with j_rand
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 6. Improved boundary repair
    lb_matrix = repmat(lb, NP, 1);
    ub_matrix = repmat(ub, NP, 1);
    
    % Reflection with damping for out-of-bounds solutions
    below_lb = offspring < lb_matrix;
    above_ub = offspring > ub_matrix;
    
    offspring(below_lb) = lb_matrix(below_lb) + 0.5*(popdecs(below_lb) - lb_matrix(below_lb));
    offspring(above_ub) = ub_matrix(above_ub) - 0.5*(ub_matrix(above_ub) - popdecs(above_ub));
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb_matrix), ub_matrix);
end
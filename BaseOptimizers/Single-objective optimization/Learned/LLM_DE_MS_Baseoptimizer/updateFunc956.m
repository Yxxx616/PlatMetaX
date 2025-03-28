% MATLAB Code
function [offspring] = updateFunc956(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Feasibility-aware base vector selection
    feasible = cons <= 0;
    if any(feasible)
        [~, best_idx] = min(popfits(feasible));
        x_best = popdecs(feasible,:);
        x_best = x_best(best_idx,:);
        x_centroid = mean(popdecs(feasible,:), 1);
        alpha = 0.6;
        x_base_feas = x_best + alpha*(x_centroid - x_best);
    else
        x_base_feas = mean(popdecs, 1);
    end
    
    % Least violated solution as secondary base
    [~, lv_idx] = min(cons);
    x_lv = popdecs(lv_idx,:);
    beta = min(1, abs(cons(lv_idx))/(max(abs(cons)) + eps));
    x_base_infeas = x_lv + beta*(x_base_feas - x_lv);
    
    % Combine base vectors
    x_base = zeros(NP, D);
    for i = 1:NP
        if feasible(i)
            x_base(i,:) = x_base_feas;
        else
            x_base(i,:) = x_base_infeas;
        end
    end
    
    % 2. Compute adaptive parameters
    f_min = min(popfits); f_max = max(popfits);
    c_max = max(abs(cons));
    
    F = 0.5 + 0.5*(popfits - f_min)./(f_max - f_min + eps) .* (1 - abs(cons)./(c_max + eps));
    CR = 0.9 * (1 - abs(cons)./(c_max + eps));
    
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
    
    mutants = x_base + F.*diff1 + (1-F).*diff2;
    
    % 5. Crossover with j_rand
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 6. Boundary handling with reflection
    lb_matrix = repmat(lb, NP, 1);
    ub_matrix = repmat(ub, NP, 1);
    
    below_lb = offspring < lb_matrix;
    above_ub = offspring > ub_matrix;
    
    offspring(below_lb) = lb_matrix(below_lb) + 0.3*(popdecs(below_lb) - lb_matrix(below_lb));
    offspring(above_ub) = ub_matrix(above_ub) - 0.3*(ub_matrix(above_ub) - popdecs(above_ub));
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb_matrix), ub_matrix);
end
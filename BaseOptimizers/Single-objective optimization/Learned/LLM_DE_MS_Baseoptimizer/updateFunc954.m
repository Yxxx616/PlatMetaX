% MATLAB Code
function [offspring] = updateFunc954(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Base vector selection
    feasible = cons <= 0;
    if any(feasible)
        [~, best_idx] = min(popfits(feasible));
        x_best = popdecs(feasible,:);
        x_best = x_best(best_idx,:);
        x_centroid = mean(popdecs(feasible,:), 1);
        alpha = 0.5; % fixed blending for feasible
        x_base_feas = x_best + alpha*(x_centroid - x_best);
    else
        x_base_feas = mean(popdecs, 1);
    end
    
    [~, lv_idx] = min(cons);
    x_lv = popdecs(lv_idx,:);
    c_avg = mean(cons(cons > 0));
    beta = cons(lv_idx)/(cons(lv_idx) + c_avg + eps);
    x_base_infeas = x_lv + beta*(x_best - x_lv);
    
    % Combine base vectors
    x_base = zeros(NP, D);
    for i = 1:NP
        if cons(i) <= 0
            x_base(i,:) = x_base_feas;
        else
            x_base(i,:) = x_base_infeas;
        end
    end
    
    % 2. Compute adaptive parameters
    f_min = min(popfits); f_max = max(popfits);
    c_min = min(cons); c_max = max(cons);
    
    F1 = 0.5*(1 + (popfits - f_min)/(f_max - f_min + eps));
    F2 = 0.5*(1 - (cons - c_min)/(c_max - c_min + eps));
    CR = 0.9 - 0.5*(cons/c_max);
    
    % 3. Generate random indices (vectorized)
    all_indices = 1:NP;
    r = zeros(NP, 4);
    for i = 1:NP
        candidates = setdiff(all_indices, i);
        r(i,:) = candidates(randperm(length(candidates), 4));
    end
    
    % 4. Directional mutation
    diff1 = popdecs(r(:,1),:) - popdecs(r(:,2),:);
    diff2 = popdecs(r(:,3),:) - popdecs(r(:,4),:);
    
    mutants = x_base + ...
              F1(:, ones(1,D)).*diff1 + ...
              F2(:, ones(1,D)).*diff2;
    
    % 5. Crossover with j_rand
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 6. Boundary repair
    lb_matrix = lb(ones(NP,1),:);
    ub_matrix = ub(ones(NP,1),:);
    
    below_lb = offspring < lb_matrix;
    above_ub = offspring > ub_matrix;
    
    offspring(below_lb) = (lb_matrix(below_lb) + popdecs(below_lb))/2;
    offspring(above_ub) = (ub_matrix(above_ub) + popdecs(above_ub))/2;
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb_matrix), ub_matrix);
end
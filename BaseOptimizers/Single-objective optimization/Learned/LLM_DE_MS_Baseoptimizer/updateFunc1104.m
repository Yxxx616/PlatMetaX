% MATLAB Code
function [offspring] = updateFunc1104(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    c_max = max(abs(cons));
    
    w = (popfits - f_min) / (f_max - f_min + eps);
    c = abs(cons) / (c_max + eps);
    
    % 2. Identify best and worst solutions
    [~, best_idx] = min(popfits);
    [~, worst_idx] = max(popfits);
    x_best = popdecs(best_idx, :);
    x_worst = popdecs(worst_idx, :);
    
    % 3. Select top 30% solutions for guidance
    k = max(2, floor(NP*0.3));
    [~, sorted_fit] = sort(popfits);
    top_idx = sorted_fit(1:k);
    
    % 4. Compute fitness-guided direction
    w_top = w(top_idx);
    x_diff_f = popdecs(top_idx,:) - repmat(x_best, k, 1);
    v_f = sum(repmat(w_top,1,D) .* x_diff_f) ./ (sum(w_top) + eps);
    
    % 5. Compute constraint-guided direction
    [~, sorted_cons] = sort(abs(cons));
    cons_idx = sorted_cons(1:k);
    c_top = c(cons_idx);
    x_diff_c = popdecs(cons_idx,:) - repmat(x_worst, k, 1);
    v_c = sum(repmat(1-c_top,1,D) .* x_diff_c) ./ (sum(1-c_top) + eps);
    
    % 6. Generate random pairs for diversity
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    v_d = popdecs(r1,:) - popdecs(r2,:);
    
    % 7. Adaptive scaling factors
    F1 = 0.5 + 0.3 * (1 - w);
    F2 = 0.3 * (1 - c);
    F3 = 0.1 + 0.2 * rand(NP, 1);
    
    % 8. Mutation operation
    mutants = popdecs + F1.*repmat(v_f, NP, 1) + F2.*repmat(v_c, NP, 1) + F3.*v_d;
    
    % 9. Directional crossover
    CR = 0.85 - 0.4 * w;
    mask = rand(NP,D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % 10. Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 11. Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    
    offspring(below) = 2*lb_rep(below) - offspring(below);
    offspring(above) = 2*ub_rep(above) - offspring(above);
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb_rep), ub_rep);
end
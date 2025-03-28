% MATLAB Code
function [offspring] = updateFunc1106(popdecs, popfits, cons)
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
    
    % 3. Select top and bottom 20% solutions
    k = max(2, floor(NP*0.2));
    [~, sorted_fit] = sort(popfits);
    elite_idx = sorted_fit(1:k);
    worst_idx = sorted_fit(end-k+1:end);
    
    % 4. Compute elite-guided direction
    w_elite = w(elite_idx);
    x_diff_elite = popdecs(elite_idx,:) - repmat(x_best, k, 1);
    v_elite = sum(repmat(w_elite,1,D) .* x_diff_elite) ./ (sum(w_elite) + eps);
    
    % 5. Compute constraint-repair direction
    c_worst = c(worst_idx);
    x_diff_worst = popdecs(worst_idx,:) - repmat(x_worst, k, 1);
    v_cons = sum(repmat(1-c_worst,1,D) .* x_diff_worst) ./ (sum(1-c_worst) + eps);
    
    % 6. Generate random pairs for diversity
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    v_div = popdecs(r1,:) - popdecs(r2,:);
    
    % 7. Adaptive scaling factors
    F_fit = 0.5 * (1 - w);
    F_cons = 0.3 * c;
    F_div = 0.2 + 0.3 * rand(NP, 1);
    
    % 8. Mutation operation
    mutants = popdecs + F_fit.*repmat(v_elite, NP, 1) + F_cons.*repmat(v_cons, NP, 1) + F_div.*v_div;
    
    % 9. Directional crossover
    CR = 0.9 - 0.4 * w;
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
% MATLAB Code
function [offspring] = updateFunc1107(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    v_max = max(abs(cons));
    
    w = (popfits - f_min) / (f_max - f_min + eps);
    c = abs(cons) / (v_max + eps);
    alpha = (1 - c) .* w;
    
    % 2. Identify best and worst solutions
    [~, best_idx] = min(popfits);
    [~, worst_idx] = max(popfits);
    x_best = popdecs(best_idx, :);
    x_worst = popdecs(worst_idx, :);
    
    % 3. Select top and bottom 30% solutions
    k = max(2, floor(NP*0.3));
    [~, sorted_idx] = sort(popfits);
    elite_idx = sorted_idx(1:k);
    worst_idx = sorted_idx(end-k+1:end);
    
    % 4. Compute elite direction
    alpha_elite = alpha(elite_idx);
    x_diff_elite = popdecs(elite_idx,:) - repmat(x_best, k, 1);
    d_elite = sum(repmat(alpha_elite,1,D) .* x_diff_elite) ./ (sum(alpha_elite) + eps);
    
    % 5. Compute repair direction
    alpha_worst = alpha(worst_idx);
    x_diff_worst = popdecs(worst_idx,:) - repmat(x_worst, k, 1);
    d_repair = sum(repmat(1-alpha_worst,1,D) .* x_diff_worst) ./ (sum(1-alpha_worst) + eps);
    
    % 6. Generate random pairs for diversity
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    d_div = popdecs(r1,:) - popdecs(r2,:);
    
    % 7. Adaptive scaling factors
    F_elite = 0.4 + 0.3 * rand(NP, 1);
    F_div = 0.1 + 0.2 * rand(NP, 1);
    F_repair = 0.2 * c;
    
    % 8. Mutation operation
    mutants = popdecs + F_elite.*repmat(d_elite, NP, 1) + ...
              F_div.*d_div + F_repair.*repmat(d_repair, NP, 1);
    
    % 9. Exponential crossover
    CR = 0.85 + 0.1 * (1 - w);
    offspring = popdecs;
    for i = 1:NP
        start = randi(D);
        L = 0;
        while rand() < CR(i) && L < D
            pos = mod(start + L - 1, D) + 1;
            offspring(i,pos) = mutants(i,pos);
            L = L + 1;
        end
    end
    
    % 10. Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below = offspring < lb_rep;
    above = offspring > ub_rep;
    
    offspring(below) = 2*lb_rep(below) - offspring(below);
    offspring(above) = 2*ub_rep(above) - offspring(above);
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb_rep), ub_rep);
end
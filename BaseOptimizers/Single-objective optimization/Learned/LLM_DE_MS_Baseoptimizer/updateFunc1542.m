% MATLAB Code
function [offspring] = updateFunc1542(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    
    % 1. Calculate weights
    min_fit = min(popfits);
    max_fit = max(popfits);
    w_f = (popfits - min_fit) / (max_fit - min_fit + eps);
    
    abs_cons = abs(cons);
    max_cons = max(abs_cons);
    w_c = abs_cons / (max_cons + eps);
    
    % Combined weights (more emphasis on constraints)
    w = 0.4 * w_f + 0.6 * w_c;
    
    % 2. Select elite (top 10%)
    [~, sorted_idx] = sort(popfits);
    elite_N = max(1, floor(0.1 * NP));
    elite_pool = popdecs(sorted_idx(1:elite_N), :);
    best_sol = elite_pool(1,:);
    
    % 3. Calculate constraint-aware direction (mean of feasible solutions)
    feasible = cons <= 0;
    if any(feasible)
        mean_feasible = mean(popdecs(feasible,:), 1);
    else
        mean_feasible = best_sol;
    end
    constraint_dir = mean_feasible - popdecs;
    
    % 4. Generate random indices for differential vectors
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    mask = r1 == r2;
    while any(mask)
        r2(mask) = randi(NP, sum(mask), 1);
        mask = r1 == r2;
    end
    
    % 5. Select elite for each individual
    elite_idx = randi(elite_N, NP, 1);
    elite = elite_pool(elite_idx, :);
    elite_dir = elite - popdecs;
    
    % 6. Differential direction
    diff_dir = popdecs(r1,:) - popdecs(r2,:);
    
    % 7. Compute adaptive parameters
    F1 = 0.5 * (1 - w);
    F2 = 0.3 * w;
    F3 = 0.2 * ones(NP, 1);
    sigma = 0.1 * w;
    
    % 8. Hybrid mutation
    mutants = popdecs + F1 .* elite_dir + F2 .* constraint_dir + F3 .* diff_dir;
    mutants = mutants + sigma .* randn(NP, 1) .* randn(NP, D);
    
    % 9. Adaptive crossover
    CR = 0.1 + 0.8 * w;
    mask_cr = rand(NP, D) < CR(:, ones(1, D));
    j_rand = randi(D, NP, 1);
    mask_cr((1:NP)' + (j_rand-1)*NP) = true;
    
    offspring = popdecs;
    offspring(mask_cr) = mutants(mask_cr);
    
    % 10. Boundary handling with reflection
    mask_lb = offspring < lb;
    mask_ub = offspring > ub;
    
    offspring(mask_lb) = 2*lb(mask_lb) - offspring(mask_lb);
    offspring(mask_ub) = 2*ub(mask_ub) - offspring(mask_ub);
    
    % 11. Final boundary check
    offspring = min(max(offspring, lb), ub);
    
    % 12. Local refinement for top solutions
    top_N = max(1, round(0.05*NP));
    top_idx = sorted_idx(1:top_N);
    sigma_top = 0.02 * (ub - lb);
    offspring(top_idx,:) = offspring(top_idx,:) + sigma_top .* randn(top_N, D);
    offspring = min(max(offspring, lb), ub);
end
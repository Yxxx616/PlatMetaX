% MATLAB Code
function [offspring] = updateFunc1541(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    
    % 1. Calculate adaptive weights
    min_fit = min(popfits);
    max_fit = max(popfits);
    w_f = (popfits - min_fit) / (max_fit - min_fit + eps);
    
    abs_cons = abs(cons);
    max_cons = max(abs_cons);
    w_c = abs_cons / (max_cons + eps);
    
    % Combined weights (more emphasis on constraints)
    w = 0.3 * w_f + 0.7 * w_c;
    
    % 2. Select elite pool (top 20%)
    [~, sorted_idx] = sort(popfits);
    elite_N = max(1, floor(0.2 * NP));
    elite_pool = popdecs(sorted_idx(1:elite_N), :);
    best_sol = elite_pool(1,:);
    
    % 3. Generate random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    mask = r1 == r2;
    while any(mask)
        r2(mask) = randi(NP, sum(mask), 1);
        mask = r1 == r2;
    end
    
    % 4. Select elite for each individual
    elite_idx = randi(elite_N, NP, 1);
    elite = elite_pool(elite_idx, :);
    
    % 5. Compute adaptive parameters
    F1 = 0.5 * (1 - w);
    F2 = 0.5 * w;
    sigma = 0.2 * w;
    CR = 0.1 + 0.8 * w;
    
    % 6. Hybrid mutation
    elite_dir = elite - popdecs;
    diff_dir = popdecs(r1,:) - popdecs(r2,:);
    perturbation = sigma .* randn(NP, 1);
    
    mutants = popdecs + F1 .* elite_dir + F2 .* diff_dir;
    mutants = mutants + perturbation .* randn(NP, D);
    
    % 7. Constraint-aware refinement
    infeasible = cons > 0;
    if any(infeasible)
        refine_dir = best_sol - popdecs(infeasible,:);
        mutants(infeasible,:) = mutants(infeasible,:) + 0.5 * refine_dir .* rand(sum(infeasible), D);
    end
    
    % 8. Crossover with adaptive CR
    mask_cr = rand(NP, D) < CR(:, ones(1, D));
    j_rand = randi(D, NP, 1);
    mask_cr((1:NP)' + (j_rand-1)*NP) = true;
    
    offspring = popdecs;
    offspring(mask_cr) = mutants(mask_cr);
    
    % 9. Boundary handling with reflection
    mask_lb = offspring < lb;
    mask_ub = offspring > ub;
    
    offspring(mask_lb) = 2*lb(mask_lb) - offspring(mask_lb);
    offspring(mask_ub) = 2*ub(mask_ub) - offspring(mask_ub);
    
    % 10. Final boundary check
    offspring = min(max(offspring, lb), ub);
    
    % 11. Local search for top solutions
    top_N = max(1, round(0.1*NP));
    top_idx = sorted_idx(1:top_N);
    sigma_top = 0.05 * (ub - lb);
    offspring(top_idx,:) = offspring(top_idx,:) + sigma_top .* randn(top_N, D);
    offspring = min(max(offspring, lb), ub);
end
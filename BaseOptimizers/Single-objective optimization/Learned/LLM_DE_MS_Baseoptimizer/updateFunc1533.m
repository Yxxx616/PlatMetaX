% MATLAB Code
function [offspring] = updateFunc1533(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    
    % Normalize constraint violations (0=best, 1=worst)
    cv = max(0, cons);
    norm_cv = (cv - min(cv)) / (max(cv) - min(cv) + eps);
    
    % Fitness ranking (0=best, 1=worst)
    [~, sorted_idx] = sort(popfits);
    ranks = zeros(NP, 1);
    ranks(sorted_idx) = (0:NP-1)'/(NP-1);
    
    % Combined weights (60% fitness, 40% constraints)
    weights = 0.6*ranks + 0.4*norm_cv;
    
    % Adaptive parameters
    F = 0.2 + 0.6 * (1 - weights);
    sigma = 0.1 * (ub - lb);
    
    % Select elite from top 10%
    elite_N = max(1, round(0.1*NP));
    elite_pool = popdecs(sorted_idx(1:elite_N),:);
    elite_idx = randi(elite_N, NP, 1);
    elite_sol = elite_pool(elite_idx,:);
    
    % Generate random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    
    % Ensure distinct indices
    mask = r1 == r2;
    r2(mask) = mod(r2(mask) + randi(NP-1, sum(mask), 1), NP) + 1;
    
    % Elite-guided direction
    elite_dir = elite_sol - popdecs;
    
    % Constraint-aware perturbation
    cons_weight = 1 ./ (1 + norm_cv);
    pert_dir = popdecs(r1,:) - popdecs(r2,:);
    pert_dir = pert_dir .* cons_weight(:, ones(1,D));
    
    % Local refinement
    local_refine = sigma .* (1 - weights(:, ones(1,D))) .* randn(NP, D);
    
    % Combined mutation
    mutants = popdecs + F(:, ones(1,D)) .* elite_dir ...
              + F(:, ones(1,D)) .* pert_dir + local_refine;
    
    % Binomial crossover with adaptive CR
    CR = 0.1 + 0.7 * (1 - weights);
    mask_cr = rand(NP,D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask_cr((1:NP)' + (j_rand-1)*NP) = true;
    
    offspring = popdecs;
    offspring(mask_cr) = mutants(mask_cr);
    
    % Boundary handling with reflection
    mask_lb = offspring < lb;
    mask_ub = offspring > ub;
    offspring(mask_lb) = 2*lb(mask_lb) - offspring(mask_lb);
    offspring(mask_ub) = 2*ub(mask_ub) - offspring(mask_ub);
    
    % Final boundary check
    offspring = min(max(offspring, lb), ub);
    
    % Additional local search for top 5 solutions
    top_N = max(1, round(0.05*NP));
    top_idx = sorted_idx(1:top_N);
    sigma_top = 0.01 * (ub - lb);
    offspring(top_idx,:) = offspring(top_idx,:) + sigma_top .* randn(top_N, D);
    offspring = min(max(offspring, lb), ub);
end
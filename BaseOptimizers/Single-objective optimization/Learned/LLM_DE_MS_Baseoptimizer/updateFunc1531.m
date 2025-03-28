% MATLAB Code
function [offspring] = updateFunc1531(popdecs, popfits, cons)
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
    alpha = 0.6;
    weights = alpha*(1-ranks) + (1-alpha)*(1-norm_cv) + eps;
    
    % Adaptive parameters
    F = 0.2 + 0.6 * weights;
    CR = 0.1 + 0.7 * weights;
    
    % Select best from top 30%
    top_N = max(1, round(0.3*NP));
    [~, best_idx] = min(popfits(sorted_idx(1:top_N)));
    best_sol = popdecs(sorted_idx(best_idx),:);
    
    % Generate random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    
    % Ensure distinct indices
    mask = r1 == r2;
    r2(mask) = mod(r2(mask) + randi(NP-1, sum(mask), 1), NP) + 1;
    
    % Current-to-weighted-best mutation
    diff1 = repmat(best_sol, NP, 1) - popdecs;
    diff2 = popdecs(r1,:) - popdecs(r2,:);
    mutants = popdecs + F(:, ones(1,D)) .* diff1 + F(:, ones(1,D)) .* diff2;
    
    % Binomial crossover
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
    
    % Elite refinement for top 20% solutions
    top_N = max(1, round(0.2*NP));
    top_idx = sorted_idx(1:top_N);
    sigma = 0.1 * (1 - weights(top_idx, ones(1,D))) .* (ub - lb);
    offspring(top_idx,:) = offspring(top_idx,:) + sigma .* randn(top_N, D);
    offspring = min(max(offspring, lb), ub);
end
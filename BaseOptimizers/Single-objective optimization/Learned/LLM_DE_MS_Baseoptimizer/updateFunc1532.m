% MATLAB Code
function [offspring] = updateFunc1532(popdecs, popfits, cons)
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
    
    % Combined weights (70% fitness, 30% constraints)
    alpha = 0.7;
    weights = alpha*(1-ranks) + (1-alpha)*(1-norm_cv) + eps;
    
    % Adaptive parameters
    F = 0.1 + 0.7 * weights;
    CR = 0.05 + 0.8 * weights;
    
    % Select elite from top 20%
    elite_N = max(1, round(0.2*NP));
    elite_pool = popdecs(sorted_idx(1:elite_N),:);
    elite_idx = randi(elite_N, NP, 1);
    elite_sol = elite_pool(elite_idx,:);
    
    % Generate random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    
    % Ensure distinct indices
    mask = r1 == r2;
    r2(mask) = mod(r2(mask) + randi(NP-1, sum(mask), 1), NP) + 1;
    
    % Elite-guided mutation
    diff1 = elite_sol - popdecs;
    diff2 = popdecs(r1,:) - popdecs(r2,:);
    mutants = popdecs + F(:, ones(1,D)) .* diff1 + F(:, ones(1,D)) .* diff2;
    
    % Binomial crossover
    mask_cr = rand(NP,D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask_cr((1:NP)' + (j_rand-1)*NP) = true;
    
    offspring = popdecs;
    offspring(mask_cr) = mutants(mask_cr);
    
    % Boundary handling with midpoint reflection
    mask_lb = offspring < lb;
    mask_ub = offspring > ub;
    offspring(mask_lb) = (lb(mask_lb) + popdecs(mask_lb))/2;
    offspring(mask_ub) = (ub(mask_ub) + popdecs(mask_ub))/2;
    
    % Final boundary check
    offspring = min(max(offspring, lb), ub);
    
    % Local refinement for top solutions
    top_N = max(1, round(0.1*NP));
    top_idx = sorted_idx(1:top_N);
    sigma = 0.05 * (1 - weights(top_idx, ones(1,D))) .* (ub - lb);
    offspring(top_idx,:) = offspring(top_idx,:) + sigma .* randn(top_N, D);
    offspring = min(max(offspring, lb), ub);
end
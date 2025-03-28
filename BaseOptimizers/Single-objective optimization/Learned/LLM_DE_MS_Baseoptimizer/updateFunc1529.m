% MATLAB Code
function [offspring] = updateFunc1529(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    offspring = zeros(NP, D);
    
    % Normalized constraint violations
    cv = max(0, cons);
    norm_cv = cv / (max(cv) + eps);
    
    % Fitness ranking (0=best, 1=worst)
    [~, sorted_idx] = sort(popfits);
    ranks = zeros(NP, 1);
    ranks(sorted_idx) = (0:NP-1)'/(NP-1);
    
    % Combined weights
    weights = (1 - norm_cv) .* (1 - ranks) + eps;
    
    % Adaptive parameters
    F = 0.4 + 0.4 * (1 - norm_cv);
    CR = 0.9 - 0.5 * norm_cv;
    
    % Identify best solution (feasible if possible)
    feasible_mask = cv == 0;
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
        best_sol = popdecs(feasible_mask(best_idx),:);
    else
        [~, best_idx] = min(popfits);
        best_sol = popdecs(best_idx,:);
    end
    
    % Generate random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    r3 = randi(NP, NP, 1);
    
    % Ensure distinct indices
    mask = r1 == r2;
    r2(mask) = mod(r2(mask) + randi(NP-1, sum(mask), 1) + 1;
    mask = r1 == r3 | r2 == r3;
    r3(mask) = mod(r3(mask) + randi(NP-2, sum(mask), 1) + 1;
    
    % Mutation with multiple strategies
    base = popdecs(r1,:);
    diff1 = popdecs(r2,:) - popdecs(r3,:);
    diff2 = repmat(best_sol, NP, 1) - popdecs;
    mutants = base + F(:, ones(1,D)) .* diff1 + F(:, ones(1,D)) .* diff2;
    
    % Crossover with adaptive CR
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
    
    % Elite refinement for top solutions
    top_N = max(1, round(0.1*NP));
    top_idx = sorted_idx(1:top_N);
    sigma = 0.1 * (1 - norm_cv(top_idx, ones(1,D))) .* (ub - lb);
    offspring(top_idx,:) = offspring(top_idx,:) + sigma .* randn(top_N, D);
    offspring = min(max(offspring, lb), ub);
end
% MATLAB Code
function [offspring] = updateFunc1215(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Find best individual
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
    % Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    cv_abs = abs(cons);
    max_cv = max(cv_abs);
    
    % Adaptive scaling factor
    F = 0.4 * (1 + (popfits - f_min) ./ (f_max - f_min + eps)) .* ...
        (1 - cv_abs ./ (max_cv + eps));
    F = F(:, ones(1, D));
    
    % Mutation
    idx = (1:NP)';
    r1 = mod(idx + randi(NP-1, NP, 1), NP) + 1;
    r2 = mod(idx + randi(NP-1, NP, 1), NP) + 1;
    r3 = mod(idx + randi(NP-1, NP, 1), NP) + 1;
    r4 = mod(idx + randi(NP-1, NP, 1), NP) + 1;
    
    diff1 = popdecs(r1,:) - popdecs(r2,:);
    diff2 = popdecs(r3,:) - popdecs(r4,:);
    mutants = x_best(ones(NP,1), :) + F .* diff1 + 0.5 * F .* diff2;
    
    % Dynamic crossover rate
    [~, rank_idx] = sort(popfits);
    ranks = zeros(NP, 1);
    ranks(rank_idx) = 1:NP;
    CR = 0.2 + 0.6 * (1 - ranks/NP) .* (1 - cv_abs/(max_cv + eps));
    CR = CR(:, ones(1, D));
    
    % Crossover
    mask = rand(NP, D) < CR;
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling with reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = 2*lb(lb_mask) - offspring(lb_mask);
    offspring(ub_mask) = 2*ub(ub_mask) - offspring(ub_mask);
    
    % Final bounds check
    offspring = min(max(offspring, lb), ub);
end
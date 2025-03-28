% MATLAB Code
function [offspring] = updateFunc1216(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Calculate weighted best individual
    eps_val = 1e-10;
    weights = 1./(abs(popfits) + abs(cons) + eps_val);
    x_wbest = sum(popdecs .* weights, 1) ./ sum(weights);
    
    % Normalize fitness and constraints
    f_min = min(popfits);
    f_max = max(popfits);
    cv_abs = abs(cons);
    max_cv = max(cv_abs);
    
    % Adaptive scaling factor with direction awareness
    F_base = 0.5;
    F = F_base ./ (1 + (popfits - f_min) ./ (f_max - f_min + eps_val)) .* ...
        (1 - cv_abs ./ (max_cv + eps_val));
    F = F(:, ones(1, D));
    
    % Mutation with direction awareness
    idx = (1:NP)';
    r1 = mod(idx + randi(NP-1, NP, 1), NP) + 1;
    r2 = mod(idx + randi(NP-1, NP, 1), NP) + 1;
    r3 = mod(idx + randi(NP-1, NP, 1), NP) + 1;
    r4 = mod(idx + randi(NP-1, NP, 1), NP) + 1;
    
    diff1 = popdecs(r1,:) - popdecs(r2,:);
    diff2 = popdecs(r3,:) - popdecs(r4,:);
    dir_sign = sign(popfits(r3) - popfits(r4));
    mutants = x_wbest(ones(NP,1), :) + F .* diff1 + 0.5 * F .* diff2 .* dir_sign(:, ones(1, D));
    
    % Dynamic crossover rate based on rank and constraints
    [~, rank_idx] = sort(popfits);
    ranks = zeros(NP, 1);
    ranks(rank_idx) = 1:NP;
    CR = 0.1 + 0.7 * (1 - ranks/NP) .* (1 - cv_abs/(max_cv + eps_val));
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
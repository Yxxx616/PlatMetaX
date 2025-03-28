% MATLAB Code
function [offspring] = updateFunc1229(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps_val = 1e-10;
    
    % 1. Elite selection considering constraints
    penalty = popfits + 1000*max(0, cons); % Large penalty for constraints
    [~, elite_idx] = min(penalty);
    x_elite = popdecs(elite_idx, :);
    
    % 2. Generate four distinct random indices for each target
    idx = 1:NP;
    R = zeros(NP, 4);
    for i = 1:NP
        available = setdiff(idx, i);
        R(i,:) = available(randperm(length(available), 4));
    end
    r1 = R(:,1); r2 = R(:,2); r3 = R(:,3); r4 = R(:,4);
    
    % 3. Calculate direction vectors
    elite_dir = x_elite(ones(NP,1), :) - popdecs;
    diff_dir1 = popdecs(r1,:) - popdecs(r2,:);
    diff_dir2 = popdecs(r3,:) - popdecs(r4,:);
    
    % Adaptive weights based on constraint violation
    cv_abs = abs(cons);
    max_cv = max(cv_abs);
    beta1 = 0.5 + 0.3*(1 - cv_abs/(max_cv + eps_val));
    beta2 = 0.3 + 0.2*rand(NP,1);
    beta3 = 0.2 + 0.1*rand(NP,1);
    
    directions = beta1.*elite_dir + beta2.*diff_dir1 + beta3.*diff_dir2;
    
    % 4. Adaptive scaling factors
    f_min = min(popfits);
    f_max = max(popfits);
    F = 0.4 + 0.3*(cv_abs./(max_cv + eps_val)) + ...
        0.3*((popfits - f_min)./(f_max - f_min + eps_val));
    F = F(:, ones(1, D));
    
    % 5. Mutation with small perturbation
    noise = 0.05 * (ub - lb) .* randn(NP, D);
    mutants = popdecs + F .* directions + noise;
    
    % 6. Rank-based adaptive crossover
    [~, rank_idx] = sort(popfits);
    ranks = zeros(NP, 1);
    ranks(rank_idx) = 1:NP;
    CR = 0.9 * (1 - ranks/NP) .* (1 - cv_abs/(max_cv + eps_val));
    CR = CR(:, ones(1, D));
    
    % Perform crossover
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
% MATLAB Code
function [offspring] = updateFunc1045(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Find best individual
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
    % Select top 20% as elites
    [~, sorted_idx] = sort(popfits);
    elite_num = max(2, floor(0.2*NP));
    elite_pool = popdecs(sorted_idx(1:elite_num), :);
    
    % Normalize fitness and constraints
    f_min = min(popfits); f_max = max(popfits);
    c_min = min(cons); c_max = max(cons);
    norm_f = (popfits - f_min) / (f_max - f_min + eps);
    norm_c = (cons - c_min) / (c_max - c_min + eps);
    
    % Calculate weights combining fitness and constraints
    f_mean = mean(popfits); f_std = std(popfits);
    c_mean = mean(cons); c_std = std(cons);
    alpha = 2; beta = 2;
    w = 1./(1 + exp(alpha*(popfits-f_mean)/f_std)) .* ...
        1./(1 + exp(beta*(cons-c_mean)/c_std));
    w = w ./ sum(w);
    
    % Compute direction vectors (vectorized)
    diff = reshape(elite_pool, elite_num, 1, D) - reshape(popdecs, 1, NP, D);
    weighted_diff = sum(reshape(w(1:elite_num), elite_num, 1, 1) .* diff, 1);
    norm_diff = sqrt(sum(weighted_diff.^2, 3)) + eps;
    direction = weighted_diff ./ reshape(norm_diff, NP, 1, D);
    direction = squeeze(direction)';
    
    % Adaptive parameters
    F_base = 0.5;
    lambda = 0.4;
    F = F_base * (1 + lambda * tanh(norm_c));
    eta = 0.1 * (1 - norm_f);
    
    % Mutation (vectorized)
    mutants = popdecs + F .* direction + eta .* (x_best - popdecs);
    
    % Dynamic crossover
    CR_base = 0.9;
    CR = CR_base * (1 - norm_f);
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR(:,ones(1,D));
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling - reflection
    lb_viol = offspring < lb;
    ub_viol = offspring > ub;
    offspring(lb_viol) = 2*lb(lb_viol) - offspring(lb_viol);
    offspring(ub_viol) = 2*ub(ub_viol) - offspring(ub_viol);
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub), lb);
end
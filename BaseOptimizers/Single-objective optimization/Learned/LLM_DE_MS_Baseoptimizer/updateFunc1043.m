% MATLAB Code
function [offspring] = updateFunc1043(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Find best individual (min fitness)
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
    % Select top 20% as elites
    [~, sorted_idx] = sort(popfits);
    elite_num = max(2, floor(0.2*NP));
    elite_pool = popdecs(sorted_idx(1:elite_num), :);
    
    % Normalize constraint violations [0,1]
    min_c = min(cons);
    max_c = max(cons);
    norm_cons = (cons - min_c) / (max_c - min_c + eps);
    
    % Normalize fitness values [0,1]
    min_f = min(popfits);
    max_f = max(popfits);
    norm_fits = (popfits - min_f) / (max_f - min_f + eps);
    
    % Adaptive scaling factors
    F = 0.4 + 0.5 * norm_cons;  % More violation -> larger step
    
    % Select random elite pairs
    e1 = randi(elite_num, NP, 1);
    e2 = mod(e1 + randi(elite_num-1, NP, 1) + 1;
    
    % Base mutation vectors
    base_mutants = popdecs + F .* (elite_pool(e1,:) - elite_pool(e2,:));
    
    % Add fitness-based perturbation
    lambda = 0.5 * (1 - norm_fits);
    perturbation = lambda .* (x_best - popdecs);
    mutants = base_mutants + perturbation;
    
    % Dynamic crossover rate
    CR = 0.85 - 0.35 * norm_fits;  % Better fitness -> lower CR
    j_rand = randi(D, NP, 1);
    mask = rand(NP,D) < CR(:,ones(1,D));
    mask(sub2ind([NP,D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling with adaptive method
    lb_viol = offspring < lb;
    ub_viol = offspring > ub;
    
    % Reflection for 70% of violations
    reflect = rand(NP,D) < 0.7;
    offspring(lb_viol & reflect) = 2*lb(lb_viol & reflect) - offspring(lb_viol & reflect);
    offspring(ub_viol & reflect) = 2*ub(ub_viol & reflect) - offspring(ub_viol & reflect);
    
    % Random reinitialization for remaining 30%
    rand_mask = (lb_viol | ub_viol) & ~reflect;
    offspring(rand_mask) = lb(rand_mask) + (ub(rand_mask)-lb(rand_mask)).*rand(sum(rand_mask(:)),1);
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub), lb);
end
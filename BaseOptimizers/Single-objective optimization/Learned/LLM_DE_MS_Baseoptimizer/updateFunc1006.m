% MATLAB Code
function [offspring] = updateFunc1006(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize fitness and constraints
    norm_fits = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    norm_cons = (cons - min(cons)) / (max(cons) - min(cons) + eps);
    
    % Weighted elite selection
    avg_fit = mean(popfits);
    avg_cons = mean(cons);
    weights = 1./(1 + exp(popfits - avg_fit)) .* 1./(1 + exp(cons - avg_cons));
    x_elite = (weights' * popdecs) / sum(weights);
    
    % Select best feasible solution
    feasible = cons <= 0;
    if any(feasible)
        [~, best_idx] = min(popfits(feasible));
        x_best = popdecs(feasible,:);
        x_best = x_best(best_idx,:);
    else
        [~, best_idx] = min(cons);
        x_best = popdecs(best_idx,:);
    end
    
    % Generate random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    while any(r1 == r2 | r1 == (1:NP)' | r2 == (1:NP)')
        r1 = randi(NP, NP, 1);
        r2 = randi(NP, NP, 1);
    end
    
    % Adaptive scaling factors
    F = 0.4 + 0.3 * tanh(1 - weights) + 0.3 * tanh(1 - norm_cons);
    
    % Direction-guided mutation
    elite_term = bsxfun(@minus, x_elite, popdecs);
    diff_term = popdecs(r1,:) - popdecs(r2,:);
    best_term = bsxfun(@minus, x_best, popdecs);
    mutants = popdecs + F.*diff_term + (1-F).*best_term + 0.5*elite_term;
    
    % Constraint-aware crossover
    CR = 0.9 * (1 - norm_cons) + 0.1 * norm_fits;
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling with reflection
    lb_matrix = repmat(lb, NP, 1);
    ub_matrix = repmat(ub, NP, 1);
    
    % Reflection for boundary violations
    lb_viol = offspring < lb_matrix;
    ub_viol = offspring > ub_matrix;
    
    reflect_lb = lb_viol & (rand(NP,D) < 0.8);
    reflect_ub = ub_viol & (rand(NP,D) < 0.8);
    
    offspring(reflect_lb) = 2*lb_matrix(reflect_lb) - offspring(reflect_lb);
    offspring(reflect_ub) = 2*ub_matrix(reflect_ub) - offspring(reflect_ub);
    
    % Random replacement for remaining violations
    rand_lb = lb_viol & ~reflect_lb;
    rand_ub = ub_viol & ~reflect_ub;
    
    offspring(rand_lb) = lb_matrix(rand_lb) + rand(sum(rand_lb(:)),1).*(ub_matrix(rand_lb)-lb_matrix(rand_lb));
    offspring(rand_ub) = lb_matrix(rand_ub) + rand(sum(rand_ub(:)),1).*(ub_matrix(rand_ub)-lb_matrix(rand_ub));
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub_matrix), lb_matrix);
end
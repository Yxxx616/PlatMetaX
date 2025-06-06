% MATLAB Code
function [offspring] = updateFunc1007(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Weight calculation
    avg_fit = mean(popfits);
    avg_cons = mean(cons);
    weights = 1./(1 + exp(popfits - avg_fit)) .* 1./(1 + exp(cons - avg_cons));
    
    % Elite vector
    x_elite = (weights' * popdecs) / sum(weights);
    
    % Best feasible solution
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
    mask = (r1 == r2) | (r1 == (1:NP)') | (r2 == (1:NP)');
    while any(mask)
        r1(mask) = randi(NP, sum(mask), 1);
        r2(mask) = randi(NP, sum(mask), 1);
        mask = (r1 == r2) | (r1 == (1:NP)') | (r2 == (1:NP)');
    end
    
    % Constraint and fitness sigmoid weights
    cons_weights = 1./(1 + exp(-cons));
    fit_weights = 1./(1 + exp(-popfits));
    
    % Mutation components
    elite_term = bsxfun(@minus, x_elite, popdecs);
    diff_term = popdecs(r1,:) - popdecs(r2,:);
    diff_term = diff_term .* (1 - cons_weights(:, ones(1, D)));
    best_term = bsxfun(@minus, x_best, popdecs);
    best_term = best_term .* fit_weights(:, ones(1, D)));
    
    % Combined mutation
    F = 0.5;
    alpha = 0.3;
    mutants = popdecs + F*diff_term + (1-F)*best_term + alpha*elite_term;
    
    % Adaptive crossover
    CR = 0.9 * (1 - cons_weights) + 0.1 * fit_weights;
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling with adaptive reflection
    lb_matrix = repmat(lb, NP, 1);
    ub_matrix = repmat(ub, NP, 1);
    
    % Reflection probability based on fitness
    reflect_prob = 0.7 * fit_weights + 0.3;
    reflect_mask = rand(NP, D) < reflect_prob(:, ones(1, D));
    
    % Handle lower bound violations
    lb_viol = offspring < lb_matrix;
    reflect_lb = lb_viol & reflect_mask;
    offspring(reflect_lb) = 2*lb_matrix(reflect_lb) - offspring(reflect_lb);
    
    % Handle upper bound violations
    ub_viol = offspring > ub_matrix;
    reflect_ub = ub_viol & reflect_mask;
    offspring(reflect_ub) = 2*ub_matrix(reflect_ub) - offspring(reflect_ub);
    
    % Random replacement for remaining violations
    rand_lb = lb_viol & ~reflect_lb;
    rand_ub = ub_viol & ~reflect_ub;
    offspring(rand_lb) = lb_matrix(rand_lb) + rand(sum(rand_lb(:)),1).*(ub_matrix(rand_lb)-lb_matrix(rand_lb));
    offspring(rand_ub) = lb_matrix(rand_ub) + rand(sum(rand_ub(:)),1).*(ub_matrix(rand_ub)-lb_matrix(rand_ub));
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub_matrix), lb_matrix);
end
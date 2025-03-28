% MATLAB Code
function [offspring] = updateFunc1009(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Calculate weights
    avg_fit = mean(popfits);
    avg_cons = mean(cons);
    w_fit = 1./(1 + exp(popfits - avg_fit));
    w_cons = 1./(1 + exp(cons - avg_cons));
    
    % Elite vector (weighted average)
    x_elite = (w_fit' .* w_cons' .* popdecs) / sum(w_fit .* w_cons);
    
    % Best solution selection
    feasible = cons <= 0;
    if any(feasible)
        [~, best_idx] = min(popfits(feasible));
        x_best = popdecs(feasible(best_idx),:);
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
    
    % Mutation parameters
    F = 0.6;
    alpha = 0.4;
    beta = 0.2;
    
    % Base mutation term
    diff_term = popdecs(r1,:) - popdecs(r2,:);
    v = popdecs + F * diff_term .* w_cons(:, ones(1, D));
    
    % Best-guided term
    best_term = bsxfun(@minus, x_best, popdecs);
    
    % Elite diversity term
    elite_term = bsxfun(@minus, x_elite, popdecs);
    
    % Combined mutation
    mutants = v + alpha * best_term .* w_fit(:, ones(1, D)) + ...
              beta * elite_term;
    
    % Adaptive crossover
    CR = 0.7 * w_cons + 0.3 * (1 - w_fit);
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling with adaptive reflection
    reflect_prob = 0.6 * w_fit + 0.2;
    reflect_mask = rand(NP, D) < reflect_prob(:, ones(1, D));
    
    lb_matrix = repmat(lb, NP, 1);
    ub_matrix = repmat(ub, NP, 1);
    
    % Handle lower bound violations
    lb_viol = offspring < lb_matrix;
    reflect_lb = lb_viol & reflect_mask;
    offspring(reflect_lb) = 2*lb_matrix(reflect_lb) - offspring(reflect_lb);
    
    % Handle upper bound violations
    ub_viol = offspring > ub_matrix;
    reflect_ub = ub_viol & reflect_mask;
    offspring(reflect_ub) = 2*ub_matrix(reflect_ub) - offspring(reflect_ub);
    
    % Random replacement for remaining violations
    rand_replace = (lb_viol & ~reflect_lb) | (ub_viol & ~reflect_ub);
    offspring(rand_replace) = lb_matrix(rand_replace) + ...
        rand(sum(rand_replace(:)), 1) .* ...
        (ub_matrix(rand_replace) - lb_matrix(rand_replace));
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub_matrix), lb_matrix);
end
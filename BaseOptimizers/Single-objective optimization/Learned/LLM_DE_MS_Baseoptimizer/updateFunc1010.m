% MATLAB Code
function [offspring] = updateFunc1010(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Normalize fitness and constraints
    norm_fits = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    norm_cons = (cons - min(cons)) / (max(cons) - min(cons) + eps);
    
    % Calculate adaptive weights
    w = 1./(1 + exp(5*(norm_fits - 0.5))) .* 1./(1 + exp(5*(norm_cons - 0.5)));
    
    % Elite vector (weighted centroid)
    x_elite = sum(bsxfun(@times, popdecs, w), 1) / sum(w);
    
    % Generate random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) randi(NP-1), idx);
    r1 = r1 + (r1 >= idx);
    r2 = arrayfun(@(i) randi(NP-1), idx);
    r2 = r2 + (r2 >= idx);
    
    % Adaptive parameters
    F = 0.5 + 0.3 * randn(NP, 1);
    alpha = 0.4 * (1 - norm_cons);
    beta = 0.6 * w;
    gamma = 0.2 * (1 - w);
    
    % Base mutation with constraint-aware scaling
    diff_term = popdecs(r1,:) - popdecs(r2,:);
    v = popdecs + bsxfun(@times, F .* (1 + alpha), diff_term);
    
    % Elite guidance with constraint relaxation
    elite_term = bsxfun(@minus, x_elite, popdecs);
    
    % Random exploration term
    rand_term = gamma .* randn(NP, D);
    
    % Combined mutation
    mutants = v + bsxfun(@times, beta, elite_term) + rand_term;
    
    % Dynamic crossover rate
    CR = 0.5 + 0.3 * norm_fits + 0.2 * norm_cons;
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1, D));
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling with adaptive reflection
    reflect_prob = 0.7 * w + 0.1;
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
% MATLAB Code
function [offspring] = updateFunc854(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Identify feasible solutions
    feasible = cons <= 0;
    
    % Elite selection (feasibility-aware)
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % Best solution by fitness
    [~, best_idx] = min(popfits);
    best = popdecs(best_idx, :);
    
    % Normalize constraints for scaling factors
    c_min = min(cons);
    c_max = max(cons);
    F = 0.5 + 0.5 * (cons - c_min) ./ (c_max - c_min + eps);
    
    % Normalize fitness for weights
    f_min = min(popfits);
    f_max = max(popfits);
    W = (popfits - f_min) ./ (f_max - f_min + eps);
    
    % Generate random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    for i = 1:NP
        while r1(i) == i || r2(i) == i || r1(i) == r2(i)
            r1(i) = randi(NP);
            r2(i) = randi(NP);
        end
    end
    
    % Direction vectors
    d1 = bsxfun(@minus, elite, popdecs);
    d2 = popdecs(r1,:) - popdecs(r2,:);
    d3 = bsxfun(@minus, best, popdecs);
    
    % Weighted mutation
    mutant = popdecs + bsxfun(@times, F, d1) + ...
             bsxfun(@times, (1-F), bsxfun(@times, W, d2) + bsxfun(@times, (1-W), d3));
    
    % Adaptive crossover rate
    CR = 0.9 * W;
    
    % Binomial crossover
    mask = rand(NP, D) < CR(:, ones(1, D));
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    offspring(below_lb) = 2*lb_rep(below_lb) - offspring(below_lb);
    offspring(above_ub) = 2*ub_rep(above_ub) - offspring(above_ub);
    offspring = max(min(offspring, ub_rep), lb_rep);
end
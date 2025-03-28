% MATLAB Code
function [offspring] = updateFunc1151(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps_val = 1e-10;
    
    % 1. Population analysis
    [~, best_idx] = min(popfits);
    [~, worst_idx] = max(popfits);
    x_best = popdecs(best_idx, :);
    x_worst = popdecs(worst_idx, :);
    
    % 2. Feasibility information
    feasible_mask = cons <= 0;
    rho = sum(feasible_mask)/NP;
    
    % 3. Compute weighted center
    if rho == 0
        center = mean(popdecs, 1);
    elseif rho < 1
        center = mean(popdecs(feasible_mask, :), 1);
    else
        center = x_best;
    end
    
    % 4. Generate random vectors for diversity
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    while any(r1 == r2)
        r2 = randi(NP, NP, 1);
    end
    x_rand = popdecs(r2, :);
    
    % 5. Compute direction vectors
    d1 = repmat(x_best - x_worst, NP, 1);
    d2 = repmat(center, NP, 1) - x_rand;
    
    % 6. Adaptive scaling factor
    c_norm = (cons - min(cons)) ./ (max(cons) - min(cons) + eps_val);
    F = 0.5 + 0.3 * tanh(5 * (1 - c_norm));
    
    % 7. Small random perturbation
    r = 0.1 * (rand(NP, D) - 0.5);
    
    % 8. Mutation
    beta = 0.5 + 0.4 * tanh(10 * (rho - 0.5));
    mutants = popdecs + F .* (d1 + beta * d2) + r;
    
    % 9. Adaptive crossover
    CR = 0.9 - 0.5 * c_norm;
    mask = rand(NP, D) < repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 10. Boundary handling with reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = 2*lb(lb_mask) - offspring(lb_mask);
    offspring(ub_mask) = 2*ub(ub_mask) - offspring(ub_mask);
    offspring = min(max(offspring, lb), ub);
end
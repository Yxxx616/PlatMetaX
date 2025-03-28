% MATLAB Code
function [offspring] = updateFunc1152(popdecs, popfits, cons)
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
    
    % 3. Generate random vectors for diversity
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    while any(r1 == r2)
        r2 = randi(NP, NP, 1);
    end
    x_rand1 = popdecs(r1, :);
    x_rand2 = popdecs(r2, :);
    
    % 4. Compute direction vectors
    d_imp = repmat(x_best - x_worst, NP, 1);
    d_div = x_rand1 - x_rand2;
    
    % 5. Adaptive scaling factor
    c_norm = (cons - min(cons)) ./ (max(cons) - min(cons) + eps_val);
    F = 0.7 * (1 - tanh(5 * c_norm)) + 0.3;
    
    % 6. Small random perturbation
    r = 0.1 * randn(NP, D);
    
    % 7. Mutation
    mutants = popdecs + F .* (d_imp + rho * d_div) + r;
    
    % 8. Adaptive crossover
    CR = 0.9 - 0.5 * rho;
    mask = rand(NP, D) < repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 9. Boundary handling with reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = 2*lb(lb_mask) - offspring(lb_mask);
    offspring(ub_mask) = 2*ub(ub_mask) - offspring(ub_mask);
    offspring = min(max(offspring, lb), ub);
end
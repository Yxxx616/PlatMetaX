% MATLAB Code
function [offspring] = updateFunc1153(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Identify elite solutions
    [~, sorted_idx] = sort(popfits);
    best_idx = sorted_idx(1);
    x_best = popdecs(best_idx, :);
    
    % 2. Create weighted combination of top solutions
    top_num = max(3, round(0.2*NP));
    top_idx = sorted_idx(1:top_num);
    weights = exp(-linspace(0,5,top_num))';
    weights = weights/sum(weights);
    x_top = weights' * popdecs(top_idx, :);
    
    % 3. Calculate feasibility ratio
    feasible_mask = cons <= 0;
    rho = sum(feasible_mask)/NP;
    
    % 4. Generate random indices
    r = randi(NP, NP, 1);
    while any(r == (1:NP)')
        r = randi(NP, NP, 1);
    end
    x_rand = popdecs(r, :);
    
    % 5. Adaptive scaling factors
    F1 = 0.5 * (1 + tanh(5*rho));
    cons_norm = abs(cons) ./ (max(abs(cons)) + 1e-10);
    F2 = 0.3 * (1 - exp(-5*cons_norm));
    
    % 6. Small perturbation
    epsilon = 0.1 * randn(NP, D);
    
    % 7. Mutation
    mutants = popdecs + ...
        F1 .* (repmat(x_best, NP, 1) - popdecs) + ...
        F2 .* (repmat(x_top, NP, 1) - x_rand) + ...
        epsilon;
    
    % 8. Crossover with adaptive CR
    CR = 0.9 - 0.4*(1-rho);
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
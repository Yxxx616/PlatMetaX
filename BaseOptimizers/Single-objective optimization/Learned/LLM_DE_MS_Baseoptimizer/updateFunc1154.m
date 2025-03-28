% MATLAB Code
function [offspring] = updateFunc1154(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Identify elite solutions
    [~, sorted_idx] = sort(popfits);
    best_idx = sorted_idx(1);
    x_best = popdecs(best_idx, :);
    
    % 2. Calculate feasibility information
    feasible_mask = cons <= 0;
    rho = sum(feasible_mask)/NP; % feasibility ratio
    feasible_pop = popdecs(feasible_mask, :);
    if isempty(feasible_pop)
        feasible_pop = popdecs;
    end
    
    % 3. Generate random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    while any(r1 == (1:NP)')
        r1 = randi(NP, NP, 1);
    end
    while any(r2 == (1:NP)' | r2 == r1)
        r2 = randi(NP, NP, 1);
    end
    
    % 4. Select random feasible solutions
    feasible_idx = randi(size(feasible_pop,1), NP, 1);
    x_feas = feasible_pop(feasible_idx, :);
    
    % 5. Adaptive scaling factors
    F1 = 0.8 * (1 - rho);
    F2 = 0.5 * rho;
    F3 = 0.3 * (1 - rho);
    
    % 6. Feasibility weights
    cons_norm = abs(cons) ./ (max(abs(cons)) + 1e-10);
    w_f = 1 - cons_norm;
    
    % 7. Mutation
    epsilon = 0.1 * randn(NP, D);
    mutants = popdecs + ...
        F1 * (repmat(x_best, NP, 1) - popdecs) + ...
        F2 * w_f .* (x_feas - popdecs(r1,:)) + ...
        F3 * (popdecs(r1,:) - popdecs(r2,:)) + ...
        epsilon .* (1 + repmat(cons_norm, 1, D));
    
    % 8. Crossover with adaptive CR
    CR = 0.9 - 0.5*(1-rho);
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
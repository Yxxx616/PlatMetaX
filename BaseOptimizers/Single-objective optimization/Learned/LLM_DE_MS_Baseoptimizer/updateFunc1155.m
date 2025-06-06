% MATLAB Code
function [offspring] = updateFunc1155(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Identify elite solutions (top 30%)
    [~, sorted_idx] = sort(popfits);
    elite_size = max(1, floor(0.3*NP));
    elites = popdecs(sorted_idx(1:elite_size), :);
    x_best = elites(1,:);
    
    % 2. Calculate feasibility information
    feasible_mask = cons <= 0;
    rho = sum(feasible_mask)/NP;
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
    
    % 4. Select random feasible and elite solutions
    feasible_idx = randi(size(feasible_pop,1), NP, 1);
    x_feas = feasible_pop(feasible_idx, :);
    elite_idx = randi(elite_size, NP, 1);
    x_elite = elites(elite_idx, :);
    
    % 5. Adaptive weights
    w_best = 0.5 * (1 - rho);
    w_feas = 0.3 * rho;
    w_diverse = 0.2;
    
    % 6. Constraint-aware mutation
    cons_norm = abs(cons) ./ (max(abs(cons)) + 1e-10);
    epsilon = 0.1 * randn(NP, D);
    mutants = popdecs + ...
        w_best * (repmat(x_best, NP, 1) - popdecs) + ...
        w_feas * (x_feas - popdecs) + ...
        w_diverse * (popdecs(r1,:) - popdecs(r2,:)) + ...
        epsilon .* (1 + repmat(cons_norm, 1, D));
    
    % 7. Elite-guided crossover
    CR = 0.9 - 0.4*rho;
    mask = rand(NP, D) < repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 8. Boundary handling with reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = 2*lb(lb_mask) - offspring(lb_mask);
    offspring(ub_mask) = 2*ub(ub_mask) - offspring(ub_mask);
    offspring = min(max(offspring, lb), ub);
end
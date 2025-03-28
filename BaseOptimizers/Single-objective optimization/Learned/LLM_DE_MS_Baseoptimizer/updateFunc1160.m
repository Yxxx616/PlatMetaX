% MATLAB Code
function [offspring] = updateFunc1160(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Identify elite solutions (top 30%)
    [~, sorted_idx] = sort(popfits, 'descend');
    elite_size = max(1, floor(0.3*NP));
    elites = popdecs(sorted_idx(1:elite_size), :);
    
    % 2. Calculate feasibility information
    feasible_mask = cons <= 0;
    rho = sum(feasible_mask)/NP;
    feasible_pop = popdecs(feasible_mask, :);
    if isempty(feasible_pop)
        feasible_pop = popdecs;
    end
    
    % 3. Generate unique random indices
    r1 = randi(NP, NP, 1);
    while any(r1 == (1:NP)')
        r1 = randi(NP, NP, 1);
    end
    r2 = randi(NP, NP, 1);
    while any(r2 == (1:NP)' | r2 == r1)
        r2 = randi(NP, NP, 1);
    end
    
    % 4. Select random elite and feasible solutions
    elite_idx = randi(elite_size, NP, 1);
    x_elite = elites(elite_idx, :);
    feasible_idx = randi(size(feasible_pop,1), NP, 1);
    x_feas = feasible_pop(feasible_idx, :);
    
    % 5. Calculate enhanced feasibility weights
    abs_cons = abs(cons);
    max_cons = max(abs_cons) + 1e-10;
    w_feas = exp(-abs_cons./max_cons);  % Exponential weighting
    
    % 6. Dynamic scaling factors
    F1 = 0.7 * (1 - rho) + 0.3;
    F2 = 0.5 * rho + 0.3;
    F3 = 0.4 * (1 - rho);
    
    % 7. Enhanced mutation with triple guidance
    mutants = popdecs + ...
        F1 * (x_elite - popdecs) + ...
        F2 * (x_feas - popdecs) .* w_feas + ...
        F3 * (popdecs(r1,:) - popdecs(r2,:)) .* (1-w_feas);
    
    % 8. Adaptive crossover
    CR = 0.95 - 0.4*rho;
    mask = rand(NP, D) < CR;
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 9. Improved boundary handling with reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = 2*lb(lb_mask) - offspring(lb_mask);
    offspring(ub_mask) = 2*ub(ub_mask) - offspring(ub_mask);
    
    % Ensure final bounds
    offspring = min(max(offspring, lb), ub);
end
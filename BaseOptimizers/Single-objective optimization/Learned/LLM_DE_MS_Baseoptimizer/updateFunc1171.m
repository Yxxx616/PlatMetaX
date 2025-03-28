% MATLAB Code
function [offspring] = updateFunc1171(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Identify elite solutions (top 20%)
    [~, sorted_idx] = sort(popfits, 'descend');
    elite_size = max(1, floor(0.2*NP));
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
    r2 = randi(NP, NP, 1);
    while any(r1 == (1:NP)')
        r1 = randi(NP, NP, 1);
    end
    while any(r2 == (1:NP)' | r2 == r1)
        r2 = randi(NP, NP, 1);
    end
    
    % 4. Select random elite and feasible solutions
    elite_idx = randi(elite_size, NP, 1);
    x_elite = elites(elite_idx, :);
    feasible_idx = randi(size(feasible_pop,1), NP, 1);
    x_feas = feasible_pop(feasible_idx, :);
    
    % 5. Enhanced feasibility weights with exponential transformation
    abs_cons = abs(cons);
    max_cons = max(abs_cons) + 1e-10;
    w_feas = exp(-abs_cons./max_cons);
    
    % 6. Adaptive scaling factors
    F_elite = 0.8 - 0.4*rho;
    F_feas = 0.6*(1-rho);
    F_div = 0.4*rho;
    
    % 7. Triple-component mutation with improved balance
    mutants = popdecs + ...
        F_elite * (x_elite - popdecs) + ...
        F_feas * (x_feas - popdecs) .* w_feas + ...
        F_div * (popdecs(r1,:) - popdecs(r2,:)) .* (1-w_feas);
    
    % 8. Adaptive crossover with dynamic probability
    CR = 0.85 - 0.35*rho;
    mask = rand(NP, D) < CR;
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 9. Boundary handling with adaptive reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = 2*lb(lb_mask) - offspring(lb_mask);
    offspring(ub_mask) = 2*ub(ub_mask) - offspring(ub_mask);
    
    % Final bounds check and repair
    offspring = min(max(offspring, lb), ub);
    
    % Additional random repair for extreme violations
    out_of_bounds = offspring < lb | offspring > ub;
    if any(out_of_bounds(:))
        offspring(out_of_bounds) = lb(out_of_bounds) + rand(sum(out_of_bounds(:)),1).*(ub(out_of_bounds)-lb(out_of_bounds));
    end
end
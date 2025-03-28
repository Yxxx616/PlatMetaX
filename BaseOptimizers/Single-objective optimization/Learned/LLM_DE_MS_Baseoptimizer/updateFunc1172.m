% MATLAB Code
function [offspring] = updateFunc1172(popdecs, popfits, cons)
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
    
    % 5. Enhanced feasibility weights with sigmoid transformation
    abs_cons = abs(cons);
    max_cons = max(abs_cons) + 1e-10;
    w_feas = 1./(1 + exp(5*(abs_cons./max_cons - 0.5)));
    
    % 6. Adaptive scaling factors with improved balance
    F_elite = 0.7 + 0.2*rho;
    F_feas = 0.5*(1-rho);
    F_div = 0.3*rho;
    
    % 7. Triple-component mutation with dynamic weights
    mutants = popdecs + ...
        F_elite * (x_elite - popdecs) + ...
        F_feas * (x_feas - popdecs) .* w_feas + ...
        F_div * (popdecs(r1,:) - popdecs(r2,:)) .* (1-w_feas);
    
    % 8. Adaptive crossover with dynamic probability
    CR = 0.9 - 0.4*rho;
    mask = rand(NP, D) < CR;
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 9. Boundary handling with adaptive reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = lb(lb_mask) + rand(sum(lb_mask(:)),1).*(ub(lb_mask)-lb(lb_mask));
    offspring(ub_mask) = lb(ub_mask) + rand(sum(ub_mask(:)),1).*(ub(ub_mask)-lb(ub_mask));
    
    % Final bounds check
    offspring = min(max(offspring, lb), ub);
end
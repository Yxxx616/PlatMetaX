% MATLAB Code
function [offspring] = updateFunc1183(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Elite selection (top 20%)
    [~, sorted_idx] = sort(popfits, 'descend');
    elite_size = max(1, floor(0.2*NP));
    elites = popdecs(sorted_idx(1:elite_size), :);
    
    % 2. Feasibility information
    feasible_mask = cons <= 0;
    rho = sum(feasible_mask)/NP;
    feasible_pop = popdecs(feasible_mask, :);
    if isempty(feasible_pop)
        feasible_pop = popdecs;
    end
    
    % 3. Generate unique random indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    mask_same = (r1 == (1:NP)') | (r2 == (1:NP)') | (r1 == r2);
    while any(mask_same)
        r1(mask_same) = randi(NP, sum(mask_same), 1);
        r2(mask_same) = randi(NP, sum(mask_same), 1);
        mask_same = (r1 == (1:NP)') | (r2 == (1:NP)') | (r1 == r2);
    end
    
    % 4. Select random elite and feasible solutions
    elite_idx = randi(elite_size, NP, 1);
    x_elite = elites(elite_idx, :);
    feasible_idx = randi(size(feasible_pop,1), NP, 1);
    x_feas = feasible_pop(feasible_idx, :);
    
    % 5. Enhanced feasibility weights
    abs_cons = abs(cons);
    max_cons = max(abs_cons) + 1e-10;
    w = 1./(1 + exp(-5*(1 - abs_cons./max_cons)));
    
    % 6. Improved adaptive scaling factors
    F_elite = 0.8 - 0.3*rho;
    F_feas = 0.6*rho.^0.7;
    F_div = 0.4*(1-rho).^0.6;
    
    % 7. Three-component mutation with adaptive weights
    mutants = popdecs + ...
        F_elite * (x_elite - popdecs) + ...
        F_feas * (x_feas - popdecs) .* w + ...
        F_div * (popdecs(r1,:) - popdecs(r2,:)) .* (1-w);
    
    % 8. Adaptive crossover with fitness-based CR
    CR = 0.95 - 0.45*w;
    mask = rand(NP, D) < CR;
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 9. Boundary handling with reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = min(ub(lb_mask), 2*lb(lb_mask) - offspring(lb_mask));
    offspring(ub_mask) = max(lb(ub_mask), 2*ub(ub_mask) - offspring(ub_mask));
end
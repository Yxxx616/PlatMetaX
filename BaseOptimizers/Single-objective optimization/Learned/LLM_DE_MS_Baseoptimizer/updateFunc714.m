% MATLAB Code
function [offspring] = updateFunc714(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    iter = 714;  % Current iteration number
    
    % Normalize fitness and constraints
    f_min = min(popfits); f_max = max(popfits);
    c_min = min(cons); c_max = max(cons);
    
    w_f = 1 + (popfits - f_min) / (f_max - f_min + eps);
    w_c = 1 + (cons - c_min) / (c_max - c_min + eps);
    
    % Identify elite individual (best feasible or least infeasible)
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % Generate random indices
    idx = arrayfun(@(x) randperm(NP, 4), 1:NP, 'UniformOutput', false);
    idx = cell2mat(idx');
    r1 = idx(:,1); r2 = idx(:,2); 
    r3 = idx(:,3); r4 = idx(:,4);
    
    % Dynamic scaling factors
    F1 = 0.5 * (1 + sin(pi * iter / 200));
    F2 = 0.3 * (1 + cos(pi * iter / 300));
    
    % Create mutation vectors
    mutant = repmat(elite, NP, 1) + ...
             F1 * (popdecs(r1,:) - popdecs(r2,:)) .* w_f + ...
             F2 * (popdecs(r3,:) - popdecs(r4,:)) .* w_c;
    
    % Adaptive crossover rate
    CR = 0.5 + 0.3 * sin(pi * (w_f + w_c) / 4);
    CR = min(max(CR, 0.1), 0.9);
    
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
    
    % Final clipping to ensure feasibility
    offspring = max(min(offspring, ub_rep), lb_rep);
end
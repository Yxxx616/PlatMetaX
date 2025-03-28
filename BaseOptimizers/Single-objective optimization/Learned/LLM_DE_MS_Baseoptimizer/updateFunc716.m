% MATLAB Code
function [offspring] = updateFunc716(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    iter = 716;
    max_iter = 1000;
    
    % Normalize fitness and constraints
    f_min = min(popfits); f_max = max(popfits);
    c_min = min(cons); c_max = max(cons);
    
    w_f = (popfits - f_min) / (f_max - f_min + eps);
    w_c = (cons - c_min) / (c_max - c_min + eps);
    
    % Identify elite individual (best feasible or least infeasible)
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % Generate random indices (6 per individual)
    idx = arrayfun(@(x) randperm(NP, 6), 1:NP, 'UniformOutput', false);
    idx = cell2mat(idx');
    r1 = idx(:,1); r2 = idx(:,2); r3 = idx(:,3);
    r4 = idx(:,4); r5 = idx(:,5); r6 = idx(:,6);
    
    % Adaptive scaling factor
    F = 0.5 + 0.3 * sin(pi * iter/max_iter);
    
    % Create mutation vectors
    base = repmat(elite, NP, 1);
    diff1 = popdecs(r1,:) - popdecs(r2,:);
    diff_f = (popdecs(r3,:) - popdecs(r4,:)) .* w_f(:, ones(1, D));
    diff_c = (popdecs(r5,:) - popdecs(r6,:)) .* w_c(:, ones(1, D));
    
    mutant = base + F * diff1 + F/2 * diff_f + F/2 * diff_c;
    
    % Adaptive crossover rate
    CR = 0.5 + 0.2 * cos(pi * (w_f + w_c)/4);
    CR = min(max(CR, 0.1), 0.9);
    
    % Binomial crossover
    mask = rand(NP, D) < CR(:, ones(1, D));
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % Boundary handling with bounce-back
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    offspring(below_lb) = lb_rep(below_lb) + rand(sum(below_lb(:)),1) .* ...
                         (popdecs(below_lb) - lb_rep(below_lb));
    offspring(above_ub) = ub_rep(above_ub) - rand(sum(above_ub(:)),1) .* ...
                         (ub_rep(above_ub) - popdecs(above_ub));
    
    % Final clipping
    offspring = max(min(offspring, ub_rep), lb_rep);
end
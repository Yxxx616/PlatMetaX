% MATLAB Code
function [offspring] = updateFunc731(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Elite selection with feasibility consideration
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % Normalize fitness and constraints
    f_min = min(popfits); f_max = max(popfits);
    c_min = min(cons); c_max = max(cons);
    
    w_f = (popfits - f_min) / (f_max - f_min + eps);
    w_c = (cons - c_min) / (c_max - c_min + eps);
    
    % Generate random indices (4 per individual)
    idx = arrayfun(@(x) randperm(NP, 4), 1:NP, 'UniformOutput', false);
    idx = cell2mat(idx');
    r1 = idx(:,1); r2 = idx(:,2); r3 = idx(:,3); r4 = idx(:,4);
    
    % Adaptive scaling factors
    F = 0.4 + 0.3 * (1 - w_c) .* (1 - w_f);
    F = repmat(F, 1, D);
    
    % Hybrid direction vectors
    d1 = popdecs(r1,:) - popdecs(r2,:);
    d2 = popdecs(r3,:) - popdecs(r4,:);
    direction = (1-w_c) .* d1 + w_c .* d2;
    
    % Elite-guided mutation with adaptive balance
    beta = 0.7 * (1 - w_f);
    beta = repmat(beta, 1, D);
    elite_rep = repmat(elite, NP, 1);
    mutant = popdecs + F .* (beta .* (elite_rep - popdecs) + (1-beta) .* direction);
    
    % Sigmoid crossover rate
    CR = 0.9 ./ (1 + exp(-5*(2*rand(NP,1)-1)));
    CR = repmat(CR, 1, D);
    
    % Crossover with guaranteed change
    mask = rand(NP, D) < CR;
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
    
    % Final projection to bounds
    offspring = max(min(offspring, ub_rep), lb_rep);
end
% MATLAB Code
function [offspring] = updateFunc712(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Normalize constraints and fitness
    c_norm = abs(cons) / (max(abs(cons)) + eps);
    f_norm = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    
    % Identify elite (best feasible) and best (overall)
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible(elite_idx), :);
    else
        [~, elite_idx] = min(c_norm);
        elite = popdecs(elite_idx, :);
    end
    [~, best_idx] = min(popfits);
    best = popdecs(best_idx, :);
    
    % Generate 6 distinct random indices for each individual
    idx = arrayfun(@(x) randperm(NP), 1:6, 'UniformOutput', false);
    r1 = idx{1}; r2 = idx{2}; r3 = idx{3}; 
    r4 = idx{4}; r5 = idx{5}; r6 = idx{6};
    
    % Adaptive scaling factor
    F = 0.5 + 0.3 * sin(pi * 712/1000);  % Oscillating between 0.2 and 0.8
    
    % Base mutation (elite-guided)
    mutant = elite + F * (popdecs(r1,:) - popdecs(r2,:)) + ...
                    F * (popdecs(r3,:) - popdecs(r4,:));
    
    % Fitness-weighted perturbation
    mutant = mutant + f_norm .* (best - popdecs);
    
    % Constraint-driven adaptation
    mutant = mutant + c_norm .* (popdecs(r5,:) - popdecs(r6,:);
    
    % Binomial crossover
    CR = 0.7 + 0.2 * rand(NP, 1);  % Random CR for each individual
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
    
    offspring(below_lb) = lb_rep(below_lb) + rand(NP, D) .* ...
                         (popdecs(below_lb) - lb_rep(below_lb));
    offspring(above_ub) = ub_rep(above_ub) - rand(NP, D) .* ...
                         (ub_rep(above_ub) - popdecs(above_ub));
    
    % Final clipping to ensure feasibility
    offspring = max(min(offspring, ub_rep), lb_rep);
end
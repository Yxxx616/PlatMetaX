% MATLAB Code
function [offspring] = updateFunc710(popdecs, popfits, cons)
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
    
    % Generate random indices
    idx = randperm(NP);
    r1 = idx(1:NP);
    r2 = idx(mod(1:NP, NP)+1);
    r3 = idx(mod(2:NP+1, NP)+1);
    
    % Adaptive parameters
    F = 0.4 + 0.3 * tanh(c_norm);  % Constraint-aware scaling
    alpha = 1./(1 + exp(-(popfits-popfits(best_idx)))); % Fitness-directed
    
    % Dual mutation strategy
    mask = rand(NP,1) < alpha;
    mutant1 = popdecs(r1,:) + F(:, ones(1,D)) .* (popdecs(r2,:) - popdecs(r3,:));
    mutant2 = best + F(:, ones(1,D)) .* (elite - best) .* alpha(:, ones(1,D));
    mutant = mask(:, ones(1,D)) .* mutant1 + (~mask(:, ones(1,D))) .* mutant2;
    
    % Adaptive crossover
    CR = 0.5 + 0.3*(1:NP)'/NP;
    mask_cr = rand(NP,D) < CR(:, ones(1,D));
    j_rand = randi(D, NP, 1);
    mask_cr = mask_cr | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask_cr) = mutant(mask_cr);
    
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
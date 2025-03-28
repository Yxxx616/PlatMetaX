% MATLAB Code
function [offspring] = updateFunc733(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Identify elite and best feasible solutions
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible(elite_idx), :);
        [~, best_feas_idx] = min(popfits(feasible));
        best_feas = popdecs(feasible(best_feas_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
        best_feas = elite;
    end
    
    % Identify worst infeasible solution
    [~, worst_infeas_idx] = max(cons(~feasible));
    if isempty(worst_infeas_idx)
        worst_infeas = elite;
    else
        infeas_indices = find(~feasible);
        worst_infeas = popdecs(infeas_indices(worst_infeas_idx), :);
    end
    
    % Normalize fitness and constraints
    f_min = min(popfits); f_max = max(popfits);
    c_min = min(cons); c_max = max(cons);
    
    w_f = (popfits - f_min) / (f_max - f_min + eps);
    w_c = (cons - c_min) / (c_max - c_min + eps);
    
    % Generate random indices (2 per individual)
    idx = arrayfun(@(x) randperm(NP, 2), 1:NP, 'UniformOutput', false);
    idx = cell2mat(idx');
    r1 = idx(:,1); r2 = idx(:,2);
    
    % Compute direction components
    elite_rep = repmat(elite, NP, 1);
    best_feas_rep = repmat(best_feas, NP, 1);
    worst_infeas_rep = repmat(worst_infeas, NP, 1);
    
    v_elite = elite_rep - popdecs;
    v_fit = (popdecs(r1,:) - popdecs(r2,:)) .* (1 + repmat(w_f, 1, D));
    v_cons = (best_feas_rep - worst_infeas_rep) .* repmat(w_c, 1, D);
    
    % Adaptive parameters
    F = 0.5 + 0.3 * cos(pi * repmat(w_f, 1, D));
    alpha = 0.6 * (1 - repmat(w_f, 1, D));
    beta = 0.4 * (1 - repmat(w_c, 1, D));
    gamma = 0.4 * repmat(w_c, 1, D);
    
    % Combined mutation
    mutant = popdecs + F .* (alpha .* v_elite + beta .* v_fit + gamma .* v_cons);
    
    % Adaptive crossover
    CR = 0.85 * (1 - 0.5 * repmat(w_c, 1, D));
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
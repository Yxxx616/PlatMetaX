% MATLAB Code
function [offspring] = updateFunc842(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Identify elite solution (best feasible or least infeasible)
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % Identify key solutions
    [~, best_idx] = min(popfits);
    [~, worst_idx] = max(popfits);
    [~, feas_idx] = min(cons);
    [~, infeas_idx] = max(cons);
    
    % Create direction vectors
    elite_rep = repmat(elite, NP, 1);
    v_elite = elite_rep - popdecs;
    
    feas_rep = repmat(popdecs(feas_idx,:), NP, 1);
    infeas_rep = repmat(popdecs(infeas_idx,:), NP, 1);
    v_feas = feas_rep - infeas_rep;
    
    best_rep = repmat(popdecs(best_idx,:), NP, 1);
    worst_rep = repmat(popdecs(worst_idx,:), NP, 1);
    v_fit = best_rep - worst_rep;
    
    % Random pairs for diversity
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    while any(r1 == r2)
        r2(r1 == r2) = randi(NP, sum(r1 == r2), 1);
    end
    v_rand = popdecs(r1,:) - popdecs(r2,:);
    
    % Normalized weights
    f_min = min(popfits); f_max = max(popfits);
    c_min = min(cons); c_max = max(cons);
    w_f = (popfits - f_min) / (f_max - f_min + eps);
    w_c = (cons - c_min) / (c_max - c_min + eps);
    
    % Enhanced adaptive parameters
    alpha = 0.8 * (1 - w_c) .* (1 + w_f);
    beta = 0.4 * w_c .* (1 - w_f);
    gamma = 0.5 * (1 + sin(pi*w_f));
    delta = 0.3;
    F = 0.7 + 0.2*(1-w_c);
    
    % Mutation with enhanced components
    mutant = popdecs + F.*(alpha.*v_elite + beta.*v_feas + gamma.*v_fit + delta.*v_rand);
    
    % Improved adaptive crossover
    CR = 0.9 - 0.4*w_f;
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
    offspring = max(min(offspring, ub_rep), lb_rep);
end
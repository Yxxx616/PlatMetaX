% MATLAB Code
function [offspring] = updateFunc858(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Identify feasible solutions
    feasible = cons <= 0;
    n_feasible = sum(feasible);
    
    % Elite selection - best feasible or least infeasible
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % Adaptive scaling factors based on constraint violation
    c_min = min(cons);
    c_max = max(cons);
    F = 0.4 + 0.4 * (cons - c_min) ./ (c_max - c_min + eps);
    
    % Fitness-based weights for feasible solutions
    if n_feasible > 0
        feasible_fits = popfits(feasible);
        w = exp(-feasible_fits) ./ sum(exp(-feasible_fits));
        d_feas = zeros(NP, D);
        for i = 1:NP
            diff = popdecs(feasible,:) - popdecs(i,:);
            d_feas(i,:) = sum(bsxfun(@times, diff, w'), 1);
        end
    else
        d_feas = zeros(NP, D);
    end
    
    % Random direction with unique indices
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    for i = 1:NP
        while r1(i) == i || r2(i) == i || r1(i) == r2(i)
            r1(i) = randi(NP);
            r2(i) = randi(NP);
        end
    end
    d_rand = popdecs(r1,:) - popdecs(r2,:);
    
    % Elite direction
    d_elite = bsxfun(@minus, elite, popdecs);
    
    % Adaptive combination factor
    alpha = 0.5 * (1 + tanh(n_feasible/NP - 0.5));
    
    % Combined mutation
    mutant = popdecs + bsxfun(@times, F, d_elite + alpha*d_feas) + ...
             bsxfun(@times, (1-F), d_rand);
    
    % Rank-based adaptive crossover rate
    [~, rank] = sort(popfits);
    CR = 0.9 - 0.5 * (rank-1)/(NP-1);
    
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
    offspring = max(min(offspring, ub_rep), lb_rep);
end
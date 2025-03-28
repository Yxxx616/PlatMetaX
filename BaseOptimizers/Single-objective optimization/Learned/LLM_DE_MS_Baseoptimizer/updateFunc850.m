% MATLAB Code
function [offspring] = updateFunc850(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    eps = 1e-12;
    
    % Elite selection
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        elite = popdecs(feasible(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % Best and worst solutions
    [~, best_idx] = min(popfits);
    [~, worst_idx] = max(popfits);
    best = popdecs(best_idx, :);
    worst = popdecs(worst_idx, :);
    
    % Feasible and infeasible means
    if any(feasible)
        feasible_mean = mean(popdecs(feasible,:), 1);
    else
        feasible_mean = mean(popdecs, 1);
    end
    if any(~feasible)
        infeasible_mean = mean(popdecs(~feasible,:), 1);
    else
        infeasible_mean = mean(popdecs, 1);
    end
    
    % Normalized weights
    f_norm = (popfits - min(popfits)) / (max(popfits) - min(popfits) + eps);
    c_norm = (cons - min(cons)) / (max(cons) - min(cons) + eps);
    
    % Adaptive parameters
    alpha = 0.5 + 0.3 * (1 - c_norm) .* (1 - f_norm);
    beta = 0.9 - 0.5 * f_norm;
    
    % Random pairs for diversity
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    while any(r1 == r2)
        r2(r1 == r2) = randi(NP, sum(r1 == r2), 1);
    end
    
    % Mutation vectors
    elite_vec = repmat(elite, NP, 1) - popdecs;
    best_worst_vec = repmat(best - worst, NP, 1);
    feasible_diff = repmat(feasible_mean - infeasible_mean, NP, 1);
    random_diff = popdecs(r1,:) - popdecs(r2,:);
    
    % Combined mutation with adaptive weights
    mutant = popdecs + alpha .* (elite_vec + beta.*best_worst_vec + ...
             (1-beta).*feasible_diff + random_diff);
    
    % Crossover with jitter
    mask = rand(NP, D) < beta(:, ones(1, D));
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
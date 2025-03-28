% MATLAB Code
function [offspring] = updateFunc618(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Elite selection considering both fitness and constraints
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        temp = find(feasible);
        elite = popdecs(temp(elite_idx), :);
    else
        [~, min_cons_idx] = min(cons);
        [~, min_fit_idx] = min(popfits);
        elite = 0.5*(popdecs(min_cons_idx,:) + 0.5*(popdecs(min_fit_idx,:));
    end
    
    % Calculate fitness-weighted centroid
    inv_fits = 1./(abs(popfits) + eps);
    weights = inv_fits/sum(inv_fits);
    centroid = weights' * popdecs;
    
    % Adaptive parameters
    [~, rank_fit] = sort(popfits);
    fit_rank = rank_fit/NP;
    F = 0.5 * (1 + fit_rank);
    
    % Generate random indices
    idx = 1:NP;
    r1 = arrayfun(@(i) randsample(setdiff(idx, i), 1), idx');
    r2 = arrayfun(@(i) randsample(setdiff(idx, [i, r1(i)]), 1), idx');
    
    % Mutation
    elite_rep = repmat(elite, NP, 1);
    centroid_rep = repmat(centroid, NP, 1);
    diff = popdecs(r1,:) - popdecs(r2,:);
    
    % Balance between elite guidance and centroid
    beta = 0.7 + 0.3*rand(NP,1);
    beta_rep = repmat(beta, 1, D);
    
    % Constraint-guided perturbation
    [~, best_cons_idx] = min(cons);
    alpha = 0.2 * (1 - exp(-abs(cons(best_cons_idx)));
    perturbation = alpha * sign(cons(best_cons_idx)) * randn(NP,D);
    
    mutant = beta_rep.*(elite_rep + F.*diff) + (1-beta_rep).*centroid_rep + perturbation;
    
    % Crossover with adaptive CR
    CR = 0.1 + 0.8*(1 - fit_rank);
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1,D)) | repmat((1:D) == j_rand, NP, 1);
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    offspring(below_lb) = min(ub_rep(below_lb), 2*lb_rep(below_lb) - offspring(below_lb));
    offspring(above_ub) = max(lb_rep(above_ub), 2*ub_rep(above_ub) - offspring(above_ub));
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb_rep), ub_rep);
end
% MATLAB Code
function [offspring] = updateFunc619(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % Elite selection with constraint handling
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        temp = find(feasible);
        elite = popdecs(temp(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % Fitness-weighted centroid calculation
    normalized_fits = popfits - min(popfits);
    weights = 1./(normalized_fits + eps);
    weights = weights/sum(weights);
    centroid = weights' * popdecs;
    
    % Adaptive scaling factors
    [~, rank_idx] = sort(popfits);
    F = 0.4 + 0.6 * (rank_idx/NP); % Higher F for worse solutions
    
    % Random indices selection
    idx = 1:NP;
    r1 = arrayfun(@(i) randsample(setdiff(idx, i), idx);
    r2 = arrayfun(@(i) randsample(setdiff(idx, [i, r1(i)]), idx);
    
    % Constraint-aware perturbation
    cons_norm = abs(cons)/max(abs(cons)+eps);
    alpha = 0.3 * cons_norm;
    perturbation = alpha .* sign(cons) .* randn(NP, D);
    
    % Mutation with adaptive balance
    beta = 0.7 + 0.3*rand(NP,1);
    beta_rep = repmat(beta, 1, D);
    elite_rep = repmat(elite, NP, 1);
    centroid_rep = repmat(centroid, NP, 1);
    diff = popdecs(r1,:) - popdecs(r2,:);
    
    mutant = beta_rep .* (elite_rep + F.*diff) + ...
             (1-beta_rep) .* centroid_rep + perturbation;
    
    % Adaptive crossover
    CR = 0.1 + 0.8*(rank_idx/NP); % Higher CR for worse solutions
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1,D)) | repmat((1:D) == j_rand, NP, 1);
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % Boundary handling with reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    offspring(below_lb) = lb_rep(below_lb) + 0.5*(ub_rep(below_lb) - offspring(below_lb));
    offspring(above_ub) = ub_rep(above_ub) - 0.5*(offspring(above_ub) - lb_rep(above_ub));
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb_rep), ub_rep);
end
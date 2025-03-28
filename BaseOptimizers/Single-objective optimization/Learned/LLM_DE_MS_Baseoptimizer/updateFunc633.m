% MATLAB Code
function [offspring] = updateFunc633(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Elite selection with constraint handling
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        temp = find(feasible);
        elite = popdecs(temp(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % 2. Ranking and probabilities
    [~, fit_rank] = sort(popfits);
    [~, cons_rank] = sort(cons);
    ranks = 0.7*fit_rank + 0.3*cons_rank; % Combined ranking
    [~, rank_order] = sort(ranks);
    probs = exp(-0.5*(rank_order/NP).^2)'; % Gaussian weights
    
    % 3. Adaptive scaling factors
    F = 0.4 + 0.5*(1 - ranks/NP).^2;
    
    % 4. Direction vectors with constraint-aware perturbation
    elite_rep = repmat(elite, NP, 1);
    alpha = 0.1 * (1 - ranks/NP);
    direction = elite_rep - popdecs + alpha.*sign(cons).*randn(NP, D);
    
    % 5. Weighted difference vectors
    delta = zeros(NP, D);
    for i = 1:NP
        candidates = setdiff(1:NP, i);
        idx = randsample(candidates, 4, true, probs(candidates));
        w1 = probs(idx(1))/(probs(idx(1))+probs(idx(2)));
        w2 = probs(idx(3))/(probs(idx(3))+probs(idx(4)));
        delta(i,:) = w1*(popdecs(idx(1),:)-popdecs(idx(2),:)) + ...
                     w2*(popdecs(idx(3),:)-popdecs(idx(4),:));
    end
    
    % 6. Combined mutation
    F_rep = repmat(F, 1, D);
    mutant = popdecs + F_rep.*direction + delta;
    
    % 7. Adaptive crossover
    CR = 0.9 - 0.5*(ranks/NP);
    CR_rep = repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR_rep | (repmat(1:D, NP, 1) == repmat(j_rand, 1, D));
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % 8. Boundary handling with midpoint reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    offspring(below_lb) = (lb_rep(below_lb) + popdecs(below_lb))/2;
    offspring(above_ub) = (ub_rep(above_ub) + popdecs(above_ub))/2;
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub_rep), lb_rep);
end
% MATLAB Code
function [offspring] = updateFunc631(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Select elite solution with constraint handling
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        temp = find(feasible);
        elite = popdecs(temp(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % 2. Calculate fitness-based weights
    [~, sorted_idx] = sort(popfits);
    ranks = zeros(NP, 1);
    ranks(sorted_idx) = 1:NP;
    weights = (NP + 1 - ranks) / sum(1:NP); % Linear rank weights
    
    % 3. Adaptive parameters
    F = 0.4 + 0.6 * (ranks / NP); % F increases with better rank
    CR = 0.9 - 0.4 * (ranks / NP); % CR decreases with better rank
    
    % 4. Generate direction vectors
    elite_rep = repmat(elite, NP, 1);
    d = elite_rep - popdecs;
    
    % 5. Generate weighted difference vectors
    delta = zeros(NP, D);
    for i = 1:NP
        % Select k=2 pairs of distinct vectors
        candidates = setdiff(1:NP, i);
        idx = randsample(candidates, 4, true, weights(candidates));
        delta(i,:) = weights(idx(1))*(popdecs(idx(1),:) - popdecs(idx(2),:)) + ...
                     weights(idx(3))*(popdecs(idx(3),:) - popdecs(idx(4),:));
    end
    
    % 6. Constraint-aware perturbation
    alpha = 0.1;
    p = alpha * sign(cons) .* randn(NP, D);
    
    % 7. Combined mutation
    F_rep = repmat(F, 1, D);
    mutant = popdecs + F_rep .* d + delta + p;
    
    % 8. Crossover with adaptive CR
    CR_rep = repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR_rep | repmat((1:D) == j_rand, NP, 1);
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % 9. Boundary handling with adaptive reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Reflection probability increases with rank
    reflect_prob = 0.3 + 0.7 * (ranks / NP);
    reflect_mask = rand(NP, D) < repmat(reflect_prob, 1, D);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    % Apply reflection
    offspring(below_lb & reflect_mask) = 2*lb_rep(below_lb & reflect_mask) - offspring(below_lb & reflect_mask);
    offspring(above_ub & reflect_mask) = 2*ub_rep(above_ub & reflect_mask) - offspring(above_ub & reflect_mask);
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub_rep), lb_rep);
end
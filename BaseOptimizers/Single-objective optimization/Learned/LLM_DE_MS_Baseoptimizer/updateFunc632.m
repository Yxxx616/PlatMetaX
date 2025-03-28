% MATLAB Code
function [offspring] = updateFunc632(popdecs, popfits, cons)
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
        [~, min_cons_idx] = min(cons);
        elite = popdecs(min_cons_idx, :);
    end
    
    % 2. Calculate combined selection probabilities (fitness + constraints)
    [~, fit_rank] = sort(popfits);
    [~, cons_rank] = sort(cons);
    ranks = 0.6*fit_rank + 0.4*cons_rank; % Combined ranking
    probs = (NP + 1 - ranks) / sum(1:NP); % Linear rank probabilities
    
    % 3. Adaptive scaling factors
    F_base = 0.5;
    F = F_base + (1 - F_base) * (ranks / NP);
    
    % 4. Generate direction vectors (elite guidance)
    elite_rep = repmat(elite, NP, 1);
    d = elite_rep - popdecs;
    
    % 5. Generate weighted difference vectors
    delta = zeros(NP, D);
    for i = 1:NP
        % Select 4 distinct vectors weighted by probs
        candidates = setdiff(1:NP, i);
        idx = datasample(candidates, 4, 'Replace', false, 'Weights', probs(candidates));
        w1 = probs(idx(1)) / (probs(idx(1)) + probs(idx(2)));
        w2 = probs(idx(3)) / (probs(idx(3)) + probs(idx(4)));
        delta(i,:) = w1*(popdecs(idx(1),:) - popdecs(idx(2),:)) + ...
                     w2*(popdecs(idx(3),:) - popdecs(idx(4),:));
    end
    
    % 6. Constraint-aware perturbation
    alpha = 0.2 * (1 - ranks/NP); % Decreases for better solutions
    p = alpha .* sign(cons) .* randn(NP, D);
    
    % 7. Combined mutation
    F_rep = repmat(F, 1, D);
    mutant = popdecs + F_rep .* d + delta + p;
    
    % 8. Crossover with adaptive probability
    CR_base = 0.9;
    CR = CR_base - 0.4 * (ranks / NP); % Better solutions use more parent info
    CR_rep = repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR_rep | repmat((1:D) == j_rand, NP, 1);
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % 9. Boundary handling with adaptive reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Reflection probability increases with rank
    reflect_prob = 0.2 + 0.8 * (ranks / NP);
    reflect_mask = rand(NP, D) < repmat(reflect_prob, 1, D);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    % Apply reflection
    offspring(below_lb & reflect_mask) = 2*lb_rep(below_lb & reflect_mask) - offspring(below_lb & reflect_mask);
    offspring(above_ub & reflect_mask) = 2*ub_rep(above_ub & reflect_mask) - offspring(above_ub & reflect_mask);
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub_rep), lb_rep);
end
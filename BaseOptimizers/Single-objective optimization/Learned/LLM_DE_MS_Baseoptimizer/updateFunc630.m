% MATLAB Code
function [offspring] = updateFunc630(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Select elite solution considering constraints
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        temp = find(feasible);
        elite = popdecs(temp(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        elite = popdecs(elite_idx, :);
    end
    
    % 2. Calculate fitness weights and ranks
    [~, sorted_idx] = sort(popfits);
    ranks = zeros(NP, 1);
    ranks(sorted_idx) = 1:NP;
    
    f_max = max(popfits);
    f_min = min(popfits);
    weights = (f_max - popfits) / (f_max - f_min + eps);
    weights = weights / sum(weights); % Normalize
    
    % 3. Adaptive parameters
    F = 0.5 * (1 + ranks / NP);
    CR = 0.9 - 0.5 * (ranks / NP);
    
    % 4. Generate direction vectors
    elite_rep = repmat(elite, NP, 1);
    d = elite_rep - popdecs;
    
    % 5. Generate weighted difference vectors
    delta = zeros(NP, D);
    for i = 1:NP
        % Select two distinct random vectors
        candidates = setdiff(1:NP, i);
        j = randsample(candidates, 1, true, weights(candidates));
        k = randsample(setdiff(candidates, j), 1, true, weights(setdiff(candidates, j)));
        
        delta(i,:) = weights(j) * (popdecs(j,:) - popdecs(k,:));
    end
    
    % 6. Constraint-aware perturbation
    alpha = 0.1;
    p = alpha * sign(cons) .* randn(NP, D);
    
    % 7. Combined mutation
    F_rep = repmat(F, 1, D);
    mutant = popdecs + F_rep .* d + delta + p;
    
    % 8. Crossover
    CR_rep = repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR_rep | repmat((1:D) == j_rand, NP, 1);
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % 9. Boundary handling with adaptive reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Reflection probability based on rank
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
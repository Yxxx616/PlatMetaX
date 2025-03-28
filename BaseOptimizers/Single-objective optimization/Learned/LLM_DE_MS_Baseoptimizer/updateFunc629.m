% MATLAB Code
function [offspring] = updateFunc629(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Elite selection with constraint handling
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        temp = find(feasible_mask);
        elite = popdecs(temp(elite_idx), :);
    else
        [~, elite_idx] = min(abs(cons));
        elite = popdecs(elite_idx, :);
    end
    
    % 2. Calculate ranks and adaptive parameters
    [~, sorted_idx] = sort(popfits);
    ranks = zeros(NP, 1);
    ranks(sorted_idx) = 1:NP;
    
    f_min = min(popfits);
    f_max = max(popfits);
    f_range = max(f_max - f_min, eps);
    weights = (popfits - f_min) / f_range;
    
    % 3. Adaptive scaling factors
    F = 0.4 + 0.6 * (ranks / NP);
    
    % 4. Select distinct random vectors
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    mask = (r1 == r2) | (r1 == (1:NP)') | (r2 == (1:NP)');
    while any(mask)
        r1(mask) = randi(NP, sum(mask), 1);
        r2(mask) = randi(NP, sum(mask), 1);
        mask = (r1 == r2) | (r1 == (1:NP)') | (r2 == (1:NP)');
    end
    
    % 5. Direction-aware mutation
    diff = popdecs(r1,:) - popdecs(r2,:);
    elite_rep = repmat(elite, NP, 1);
    F_rep = repmat(F, 1, D);
    v = elite_rep + F_rep .* diff;
    
    % 6. Constraint-aware perturbation
    alpha = 0.2;
    delta = alpha * sign(cons) .* randn(NP, D);
    v = v + delta;
    
    % 7. Fitness-guided exploration
    beta = 0.3;
    eta = beta * repmat(1 - weights, 1, D) .* randn(NP, D);
    mutant = v + eta;
    
    % 8. Adaptive crossover
    CR = 0.85 - 0.35 * (ranks / NP);
    CR_rep = repmat(CR, 1, D);
    
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR_rep | repmat((1:D) == j_rand, NP, 1);
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % 9. Boundary handling with adaptive reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Reflection probability based on fitness rank
    prob_reflect = 0.3 + 0.7 * (ranks / NP);
    prob_rep = repmat(prob_reflect, 1, D);
    
    reflect_mask = rand(NP, D) < prob_rep;
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    % Apply reflection
    offspring(below_lb & reflect_mask) = 2*lb_rep(below_lb & reflect_mask) - offspring(below_lb & reflect_mask);
    offspring(above_ub & reflect_mask) = 2*ub_rep(above_ub & reflect_mask) - offspring(above_ub & reflect_mask);
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub_rep), lb_rep);
end
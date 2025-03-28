% MATLAB Code
function [offspring] = updateFunc628(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Elite selection with improved constraint handling
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, elite_idx] = min(popfits(feasible_mask));
        temp = find(feasible_mask);
        elite = popdecs(temp(elite_idx), :);
    else
        [~, min_cons_idx] = min(abs(cons));
        elite = popdecs(min_cons_idx, :);
    end
    
    % 2. Adaptive parameters calculation
    f_min = min(popfits);
    f_max = max(popfits);
    f_range = max(f_max - f_min, eps);
    
    % 3. Direction-guided mutation with adaptive F
    ranks = tiedrank(popfits);
    F = 0.5 + 0.5 * (1 - (ranks-1)/(NP-1)); % Higher F for better solutions
    
    % Select distinct random vectors
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    mask = (r1 == r2) | (r1 == (1:NP)') | (r2 == (1:NP)');
    while any(mask)
        r1(mask) = randi(NP, sum(mask), 1);
        r2(mask) = randi(NP, sum(mask), 1);
        mask = (r1 == r2) | (r1 == (1:NP)') | (r2 == (1:NP)');
    end
    
    % Base mutation vector
    diff = popdecs(r1,:) - popdecs(r2,:);
    elite_rep = repmat(elite, NP, 1);
    F_rep = repmat(F, 1, D);
    v = elite_rep + F_rep .* diff;
    
    % 4. Constraint-aware perturbation
    alpha = 0.1;
    delta = alpha * sign(cons) .* randn(NP, D);
    
    % 5. Fitness-directed exploration
    beta = 0.2;
    fitness_weights = 1 - (popfits - f_min)/f_range;
    eta = beta * repmat(fitness_weights, 1, D) .* randn(NP, D);
    
    % Combined mutation
    mutant = v + delta + eta;
    
    % 6. Adaptive crossover
    CR_base = 0.9;
    CR = CR_base - 0.4 * (ranks-1)/(NP-1);
    CR_rep = repmat(CR, 1, D);
    
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR_rep | repmat((1:D) == j_rand, NP, 1);
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % 7. Boundary handling with adaptive reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Reflection probability based on fitness rank
    prob_reflect = 0.5 + 0.5 * (ranks-1)/(NP-1);
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

% Helper function for ranking without toolbox
function ranks = tiedrank(x)
    [~, ~, rank] = unique(x);
    counts = accumarray(rank, 1);
    sumRanks = accumarray(rank, rank, [], @sum);
    ranks = sumRanks(rank)./counts(rank);
end
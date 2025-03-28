% MATLAB Code
function [offspring] = updateFunc626(popdecs, popfits, cons)
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
        [~, min_cons_idx] = min(cons);
        [~, min_fit_idx] = min(popfits);
        elite = 0.5*(popdecs(min_cons_idx,:) + popdecs(min_fit_idx,:));
    end
    
    % 2. Adaptive parameters calculation
    f_mean = mean(popfits);
    f_std = std(popfits) + eps;
    c_max = max(abs(cons)) + eps;
    
    % Fitness-based weights (sigmoid normalized)
    weights = 1./(1 + exp(-(popfits - f_mean)./(f_std)));
    
    % Dynamic scaling factors
    F_base = 0.4 + 0.4./(1 + exp(-cons./c_max));
    alpha = 0.2;  % Fixed for simplicity
    
    % 3. Constraint-aware perturbation
    perturbation = alpha * tanh(cons) .* randn(NP, D);
    
    % 4. Differential mutation (vectorized)
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    while any(r1 == r2 | r1 == (1:NP)' | r2 == (1:NP)')
        r1 = randi(NP, NP, 1);
        r2 = randi(NP, NP, 1);
    end
    
    diff = popdecs(r1,:) - popdecs(r2,:);
    F_rep = repmat(F_base, 1, D);
    elite_rep = repmat(elite, NP, 1);
    
    % Combined mutation
    mutant1 = elite_rep + F_rep.*diff;
    mutant2 = mutant1 + perturbation;
    mutant = repmat(weights, 1, D).*mutant1 + repmat(1-weights, 1, D).*mutant2;
    
    % 5. Adaptive crossover
    [~, ranks] = sort(popfits);
    CR = 0.9 - 0.5*(ranks-1)/(NP-1);
    
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR(:, ones(1,D)) | repmat((1:D) == j_rand, NP, 1);
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % 6. Boundary handling with adaptive reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Reflection with adaptive probability
    reflect_prob = repmat(weights, 1, D);
    reflect_mask = rand(NP, D) < reflect_prob;
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    offspring(below_lb & reflect_mask) = 2*lb_rep(below_lb & reflect_mask) - offspring(below_lb & reflect_mask);
    offspring(above_ub & reflect_mask) = 2*ub_rep(above_ub & reflect_mask) - offspring(above_ub & reflect_mask);
    
    % Final bounds enforcement
    offspring = min(max(offspring, lb_rep), ub_rep);
end
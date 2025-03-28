% MATLAB Code
function [offspring] = updateFunc627(popdecs, popfits, cons)
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
        [~, elite_idx] = min(abs(cons));
        elite = popdecs(elite_idx, :);
    end
    
    % 2. Calculate adaptive parameters
    f_mean = mean(popfits);
    f_std = std(popfits) + eps;
    c_max = max(abs(cons)) + eps;
    
    % Adaptive scaling factors
    F = 0.5 * (1 + tanh((popfits - f_mean)./f_std)) .* (1 - abs(cons)./(c_max));
    F = repmat(F, 1, D);
    
    % 3. Direction-guided mutation
    % Select random vectors
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    while any(r1 == r2 | r1 == (1:NP)' | r2 == (1:NP)')
        r1 = randi(NP, NP, 1);
        r2 = randi(NP, NP, 1);
    end
    
    % Mutation with constraint-aware perturbation
    diff = popdecs(r1,:) - popdecs(r2,:);
    elite_rep = repmat(elite, NP, 1);
    perturbation = 0.1 * sign(cons) .* randn(NP, D);
    mutant = elite_rep + F .* diff + perturbation;
    
    % 4. Adaptive crossover
    [~, ranks] = sort(popfits);
    CR = 0.9 - 0.5 * (ranks-1)/(NP-1);
    CR = repmat(CR, 1, D);
    
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR | repmat((1:D) == j_rand, NP, 1);
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % 5. Boundary handling with adaptive reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    % Calculate reflection probability based on fitness
    prob_reflect = 1./(1 + exp(-(popfits - f_mean)./f_std));
    prob_reflect = repmat(prob_reflect, 1, D);
    
    reflect_mask = rand(NP, D) < prob_reflect;
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    % Apply reflection only to selected solutions
    offspring(below_lb & reflect_mask) = 2*lb_rep(below_lb & reflect_mask) - offspring(below_lb & reflect_mask);
    offspring(above_ub & reflect_mask) = 2*ub_rep(above_ub & reflect_mask) - offspring(above_ub & reflect_mask);
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub_rep), lb_rep);
end
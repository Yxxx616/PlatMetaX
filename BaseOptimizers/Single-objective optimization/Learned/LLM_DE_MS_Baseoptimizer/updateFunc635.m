% MATLAB Code
function [offspring] = updateFunc635(popdecs, popfits, cons)
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
    
    % 2. Adaptive scaling factors
    F = 0.5 + 0.3 * tanh(abs(cons));
    
    % 3. Fitness-directed mutation
    f_min = min(popfits);
    f_max = max(popfits);
    sigma_f = std(popfits);
    if sigma_f == 0, sigma_f = 1; end
    weights = exp(-(popfits - f_min)/sigma_f);
    
    mutant = zeros(NP, D);
    for i = 1:NP
        % Select donors based on fitness weights
        candidates = setdiff(1:NP, i);
        prob = weights(candidates)/sum(weights(candidates));
        idx = datasample(candidates, 2, 'Weights', prob, 'Replace', false);
        
        % Fitness-directed difference
        w = (f_max - popfits(i))/(f_max - f_min + eps);
        diff = w * (popdecs(idx(1),:) - popdecs(idx(2),:));
        
        % Mutation
        mutant(i,:) = popdecs(i,:) + F(i) * ( (elite - popdecs(i,:)) + diff );
    end
    
    % 4. Constraint-aware crossover
    [~, rank] = sort(popfits);
    CR = 0.9 * (1 - rank/NP) .* (1 - tanh(abs(cons)));
    mask = rand(NP, D) < repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = mask | (repmat(1:D, NP, 1) == repmat(j_rand, 1, D));
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % 5. Boundary handling with midpoint reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    offspring(below_lb) = (lb_rep(below_lb) + popdecs(below_lb))/2;
    offspring(above_ub) = (ub_rep(above_ub) + popdecs(above_ub))/2;
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub_rep), lb_rep);
end
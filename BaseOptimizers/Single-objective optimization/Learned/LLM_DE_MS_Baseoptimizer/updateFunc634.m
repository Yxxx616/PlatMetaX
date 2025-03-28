% MATLAB Code
function [offspring] = updateFunc634(popdecs, popfits, cons)
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
    
    % 2. Constraint-aware scaling factors
    alpha = 0.5 * tanh(abs(cons)) + 0.5;
    
    % 3. Fitness-directed weights
    f_min = min(popfits);
    sigma_f = std(popfits);
    if sigma_f == 0, sigma_f = 1; end
    weights = exp(-(popfits - f_min)/sigma_f);
    weights = weights / sum(weights);
    
    % 4. Generate mutation vectors
    F = 0.5 * ones(NP, 1);
    mutant = zeros(NP, D);
    elite_rep = repmat(elite, NP, 1);
    
    for i = 1:NP
        % Select 4 distinct random indices using weights
        candidates = setdiff(1:NP, i);
        idx = datasample(candidates, 4, 'Weights', weights(candidates), 'Replace', false);
        
        % Weighted difference vectors
        w1 = weights(idx(1)) / (weights(idx(1)) + weights(idx(2)) + eps);
        w2 = weights(idx(3)) / (weights(idx(3)) + weights(idx(4)) + eps);
        D = w1*(popdecs(idx(1),:) - popdecs(idx(2),:) + ...
            w2*(popdecs(idx(3),:) - popdecs(idx(4),:);
        
        % Constraint-aware mutation
        mutant(i,:) = popdecs(i,:) + F(i) * ( (elite - popdecs(i,:)) + alpha(i)*D );
    end
    
    % 5. Adaptive crossover
    [~, rank] = sort(popfits);
    CR = 0.9 * (1 - rank/NP);
    mask = rand(NP, D) < repmat(CR, 1, D);
    j_rand = randi(D, NP, 1);
    mask = mask | (repmat(1:D, NP, 1) == repmat(j_rand, 1, D);
    
    offspring = popdecs;
    offspring(mask) = mutant(mask);
    
    % 6. Boundary handling with midpoint reflection
    lb_rep = repmat(lb, NP, 1);
    ub_rep = repmat(ub, NP, 1);
    
    below_lb = offspring < lb_rep;
    above_ub = offspring > ub_rep;
    
    offspring(below_lb) = (lb_rep(below_lb) + popdecs(below_lb))/2;
    offspring(above_ub) = (ub_rep(above_ub) + popdecs(above_ub))/2;
    
    % Final bounds enforcement
    offspring = max(min(offspring, ub_rep), lb_rep);
end
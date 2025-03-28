% MATLAB Code
function [offspring] = updateFunc636(popdecs, popfits, cons)
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
    
    % 2. Adaptive scaling factors with sigmoid transformation
    mean_cons = mean(cons);
    std_cons = std(cons);
    if std_cons == 0, std_cons = 1; end
    F = 0.4 + 0.4./(1 + exp(-5*(cons - mean_cons)/std_cons));
    
    % 3. Fitness-directed mutation with tournament selection
    mutant = zeros(NP, D);
    sigma_f = std(popfits);
    if sigma_f == 0, sigma_f = 1; end
    
    for i = 1:NP
        % Tournament selection for donors
        candidates = setdiff(1:NP, i);
        [~, idx1] = min(popfits(randsample(candidates, 3)));
        [~, idx2] = min(popfits(randsample(candidates, 3)));
        [~, idx3] = max(popfits(randsample(candidates, 3)));
        [~, idx4] = max(popfits(randsample(candidates, 3)));
        
        % Weighted difference vectors
        w1 = exp(-popfits(idx1)/sigma_f);
        w2 = exp(-popfits(idx2)/sigma_f);
        w_sum = w1 + w2 + eps;
        diff = (w1*(popdecs(idx1,:)-popdecs(idx3,:)) + (w2*(popdecs(idx2,:)-popdecs(idx4,:)))/w_sum;
        
        % Mutation
        mutant(i,:) = popdecs(i,:) + F(i)*(elite - popdecs(i,:)) + F(i)*diff;
    end
    
    % 4. Dynamic crossover with constraint awareness
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
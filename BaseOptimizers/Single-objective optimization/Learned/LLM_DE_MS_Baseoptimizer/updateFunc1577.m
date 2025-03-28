% MATLAB Code
function [offspring] = updateFunc1577(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Sort population by fitness
    [~, sorted_idx] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(sorted_idx) = (1:NP)';
    x_best = popdecs(sorted_idx(1), :);
    
    % 2. Compute constraint-aware weights
    beta = 0.5;
    w = exp(-beta * max(0, cons));
    w = w ./ sum(w);  % Normalize
    
    % 3. Compute weighted direction vectors
    weighted_diff = zeros(NP, D);
    for i = 1:NP
        weighted_diff(i,:) = sum(bsxfun(@times, bsxfun(@minus, popdecs, popdecs(i,:)), w), 1);
    end
    
    % 4. Generate scaling factors
    F = 0.5 + 0.3 * rand(NP,1);
    
    % 5. Generate mutation vectors
    eta = 0.2 * randn(NP,1);
    offspring = popdecs + F .* weighted_diff + eta .* (x_best - popdecs);
    
    % 6. Adaptive crossover
    CR = 0.1 + 0.8 * (1 - ranks/NP);
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    offspring = popdecs .* (~mask) + offspring .* mask;
    
    % 7. Boundary handling with adaptive reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    
    % Reflection with small random perturbation
    eps = 0.2 * rand(NP,D) - 0.1;
    offspring(lb_mask) = (lb(lb_mask) + popdecs(lb_mask))/2 + eps(lb_mask);
    offspring(ub_mask) = (ub(ub_mask) + popdecs(ub_mask))/2 + eps(ub_mask);
    
    % Final boundary enforcement
    offspring = min(max(offspring, lb), ub);
end
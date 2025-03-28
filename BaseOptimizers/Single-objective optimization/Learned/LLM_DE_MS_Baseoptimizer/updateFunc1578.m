% MATLAB Code
function [offspring] = updateFunc1578(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Compute fitness weights
    max_fit = max(popfits);
    min_fit = min(popfits);
    normalized_fits = (popfits - min_fit) / (max_fit - min_fit + eps);
    w = exp(-normalized_fits);
    w = w ./ sum(w);
    
    % 2. Compute fitness-guided direction vectors
    weighted_diff = zeros(NP, D);
    for i = 1:NP
        weighted_diff(i,:) = sum(bsxfun(@times, bsxfun(@minus, popdecs, popdecs(i,:)), w);
    end
    
    % 3. Find best solution
    [~, best_idx] = min(popfits);
    x_best = popdecs(best_idx, :);
    
    % 4. Compute constraint-aware scaling factors
    alpha = 0.5 + 0.5 * tanh(max(0, cons));
    
    % 5. Generate scaling factors
    F = 0.5 + 0.3 * rand(NP,1);
    
    % 6. Generate mutation vectors
    offspring = popdecs + F .* weighted_diff + alpha .* (x_best - popdecs);
    
    % 7. Adaptive crossover
    [~, sorted_idx] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(sorted_idx) = (1:NP)';
    CR = 0.1 + 0.8 * (1 - ranks/NP);
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    offspring = popdecs .* (~mask) + offspring .* mask;
    
    % 8. Boundary handling with reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    eps = 0.1 * rand(NP,D) - 0.05;
    offspring(lb_mask) = (lb(lb_mask) + popdecs(lb_mask))/2 + eps(lb_mask);
    offspring(ub_mask) = (ub(ub_mask) + popdecs(ub_mask))/2 + eps(ub_mask);
    
    % 9. Final boundary enforcement
    offspring = min(max(offspring, lb), ub);
end
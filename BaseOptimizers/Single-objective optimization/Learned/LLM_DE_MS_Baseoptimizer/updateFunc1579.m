% MATLAB Code
function [offspring] = updateFunc1579(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Compute fitness weights with adaptive scaling
    f_min = min(popfits);
    f_max = max(popfits);
    sigma_f = max(1, (f_max - f_min)/2);
    w = exp(-(popfits - f_min)/sigma_f);
    w = w ./ sum(w);
    
    % 2. Compute weighted direction vectors (vectorized)
    pop_mean = w' * popdecs;
    weighted_diff = bsxfun(@minus, pop_mean, popdecs);
    
    % 3. Find best solution considering constraints
    feasible_mask = cons <= 0;
    if any(feasible_mask)
        [~, best_idx] = min(popfits(feasible_mask));
    else
        [~, best_idx] = min(popfits + 1e6*max(0,cons));
    end
    x_best = popdecs(best_idx, :);
    
    % 4. Compute constraint-aware scaling factors
    sigma_c = max(1, std(cons));
    alpha = 0.7 + 0.3 * tanh(max(0, cons)/sigma_c);
    
    % 5. Generate adaptive scaling factors
    F = 0.5 + 0.3 * randn(NP,1);
    
    % 6. Generate mutation vectors
    offspring = popdecs + F .* weighted_diff + alpha .* (x_best - popdecs);
    
    % 7. Rank-based adaptive crossover
    [~, sorted_idx] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(sorted_idx) = (1:NP)';
    CR = 0.2 + 0.6 * (1 - ranks/NP);
    
    % Perform crossover
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    offspring = popdecs .* (~mask) + offspring .* mask;
    
    % 8. Improved boundary handling
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    r = rand(NP,D);
    offspring(lb_mask) = lb(lb_mask) + r(lb_mask) .* (popdecs(lb_mask) - lb(lb_mask));
    offspring(ub_mask) = ub(ub_mask) - r(ub_mask) .* (ub(ub_mask) - popdecs(ub_mask));
    
    % 9. Final boundary enforcement
    offspring = min(max(offspring, lb), ub);
end
% MATLAB Code
function [offspring] = updateFunc1580(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Compute fitness weights
    f_mean = mean(popfits);
    f_std = max(std(popfits), 1e-6);
    w = 1 ./ (1 + exp((popfits - f_mean)/f_std));
    w = w ./ sum(w);
    
    % 2. Compute weighted centroid
    centroid = w' * popdecs;
    
    % 3. Find best solution considering constraints
    feasible = cons <= 0;
    if any(feasible)
        [~, best_idx] = min(popfits(feasible));
        x_best = popdecs(best_idx, :);
    else
        [~, best_idx] = min(popfits + 1e6*max(0,cons));
        x_best = popdecs(best_idx, :);
    end
    
    % 4. Compute personal best (pbest) for each individual
    [~, sorted_idx] = sort(popfits);
    pbest = popdecs(sorted_idx(1:ceil(NP/3)), :);
    pbest = pbest(randi(size(pbest,1), NP, 1), :);
    
    % 5. Compute direction vectors
    dir_vectors = zeros(NP, D);
    beta = 0.2 * rand(NP,1);
    rand_vec = randn(NP,D);
    for i = 1:NP
        if cons(i) <= 0
            dir_vectors(i,:) = x_best - popdecs(i,:);
        else
            dir_vectors(i,:) = centroid - popdecs(i,:) + beta(i)*rand_vec(i,:);
        end
    end
    
    % 6. Compute adaptive scaling factors
    c_norm = max(0, cons) ./ (max(abs(cons)) + 1e-6);
    F = 0.4 + 0.4 * tanh(abs(popfits - f_mean)/f_std) .* (1 - c_norm);
    F = min(max(F, 0.1), 0.9);
    
    % 7. Generate mutation vectors
    eta = 0.1 * rand(NP,1);
    offspring = popdecs + F.*dir_vectors + eta.*(pbest - popdecs);
    
    % 8. Rank-based crossover
    [~, ranks] = sort(popfits);
    CR = 0.1 + 0.7 * (1 - ranks/NP);
    mask = rand(NP,D) < CR(:,ones(1,D));
    j_rand = randi(D, NP, 1);
    mask((1:NP)' + (j_rand-1)*NP) = true;
    offspring = popdecs .* (~mask) + offspring .* mask;
    
    % 9. Improved boundary handling
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    r = rand(NP,D);
    offspring(lb_mask) = popdecs(lb_mask) + r(lb_mask) .* (lb(lb_mask) - popdecs(lb_mask));
    offspring(ub_mask) = popdecs(ub_mask) + r(ub_mask) .* (ub(ub_mask) - popdecs(ub_mask));
    
    % 10. Final boundary enforcement
    offspring = min(max(offspring, lb), ub);
end
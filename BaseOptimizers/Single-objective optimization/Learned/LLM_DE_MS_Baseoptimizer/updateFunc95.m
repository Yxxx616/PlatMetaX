% MATLAB Code
function [offspring] = updateFunc95(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Normalize constraints
    c_max = max(abs(cons));
    norm_cons = abs(cons) / (c_max + eps);
    
    % Rank population (lower fitness is better)
    [~, rank_order] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(rank_order) = (1:NP)';
    norm_ranks = ranks / NP;
    
    % Find best feasible individual
    [~, best_idx] = min(popfits + norm_cons);
    x_best = popdecs(best_idx, :);
    
    % Parameters
    CR = 0.85;
    F_base = 0.7;
    
    % Rank-based selection probabilities
    probs = 1./(ranks + eps);
    probs = probs / sum(probs);
    
    % Vectorized selection of r1, r2, r3
    rnd = rand(NP, 3);
    cum_probs = cumsum(probs);
    
    [~, r1] = max(rnd(:,1) <= cum_probs', [], 2);
    [~, r2] = max(rnd(:,2) <= cum_probs', [], 2);
    [~, r3] = max(rnd(:,3) <= cum_probs', [], 2);
    
    % Ensure distinct indices
    for i = 1:NP
        while r1(i) == i
            r1(i) = find(rand() <= cum_probs, 1);
        end
        while r2(i) == i || r2(i) == r1(i)
            r2(i) = find(rand() <= cum_probs, 1);
        end
        while r3(i) == i || r3(i) == r1(i) || r3(i) == r2(i)
            r3(i) = find(rand() <= cum_probs, 1);
        end
    end
    
    % Adaptive parameters
    F = F_base * (1 - norm_ranks);
    alpha = 0.3 * (1 - norm_cons);
    beta = 0.1 * norm_cons;
    
    % Constraint-aware perturbation
    xi = randn(NP, D) .* norm_cons;
    
    % Mutation
    x_r1 = popdecs(r1, :);
    x_r2 = popdecs(r2, :);
    x_r3 = popdecs(r3, :);
    
    diff = x_r2 - x_r3;
    best_diff = x_best - popdecs;
    
    v = x_r1 + F .* diff + alpha .* best_diff + beta .* xi;
    
    % Binomial crossover
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR;
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = v(mask);
end
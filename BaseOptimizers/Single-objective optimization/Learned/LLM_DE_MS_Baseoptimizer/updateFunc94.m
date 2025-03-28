% MATLAB Code
function [offspring] = updateFunc94(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    offspring = zeros(NP, D);
    
    % Normalize constraints
    c_max = max(abs(cons));
    norm_cons = abs(cons) / (c_max + eps);
    
    % Rank population based on fitness (lower is better)
    [~, rank_order] = sort(popfits);
    ranks = zeros(NP,1);
    ranks(rank_order) = (1:NP)';
    norm_ranks = ranks / NP;
    
    % Find best feasible individual (lowest fitness with minimal constraint violation)
    [~, best_idx] = min(popfits + norm_cons);
    x_best = popdecs(best_idx, :);
    
    % Parameters
    CR = 0.85;
    F_base = 0.5;
    
    % Vectorized selection of r1, r2, r3 using rank-based probabilities
    probs = 1./(ranks + eps);
    probs = probs / sum(probs);
    cum_probs = cumsum(probs);
    
    rnd = rand(NP, 3);
    r1 = zeros(NP, 1);
    r2 = zeros(NP, 1);
    r3 = zeros(NP, 1);
    
    for i = 1:NP
        % Select r1 (different from i)
        r1(i) = find(cum_probs >= rnd(i,1), 1);
        while r1(i) == i
            rnd(i,1) = rand();
            r1(i) = find(cum_probs >= rnd(i,1), 1);
        end
        
        % Select r2 (different from i and r1)
        r2(i) = find(cum_probs >= rnd(i,2), 1);
        while r2(i) == i || r2(i) == r1(i)
            rnd(i,2) = rand();
            r2(i) = find(cum_probs >= rnd(i,2), 1);
        end
        
        % Select r3 (different from i, r1, and r2)
        r3(i) = find(cum_probs >= rnd(i,3), 1);
        while r3(i) == i || r3(i) == r1(i) || r3(i) == r2(i)
            rnd(i,3) = rand();
            r3(i) = find(cum_probs >= rnd(i,3), 1);
        end
    end
    
    % Compute adaptive parameters
    F = F_base * (1 + norm_ranks);
    alpha = 0.5 * (1 - norm_cons);
    beta = 0.1 * (1 + norm_cons);
    
    % Generate constraint-aware perturbation
    xi = randn(NP, D) .* norm_cons;
    
    % Mutation
    x_r1 = popdecs(r1, :);
    x_r2 = popdecs(r2, :);
    x_r3 = popdecs(r3, :);
    
    diff = x_r2 - x_r3;
    best_diff = x_best - popdecs;
    
    % Vectorized mutation with adaptive parameters
    v = x_r1 + F .* diff + alpha .* best_diff + beta .* xi;
    
    % Binomial crossover
    j_rand = randi(D, NP, 1);
    mask = rand(NP, D) < CR;
    mask(sub2ind([NP, D], (1:NP)', j_rand)) = true;
    
    % Create offspring
    offspring = popdecs;
    offspring(mask) = v(mask);
end
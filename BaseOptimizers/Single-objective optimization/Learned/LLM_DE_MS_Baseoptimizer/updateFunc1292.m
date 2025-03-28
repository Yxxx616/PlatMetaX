% MATLAB Code
function [offspring] = updateFunc1292(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Select elite solution based on feasibility
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        temp = find(feasible);
        x_elite = popdecs(temp(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        x_elite = popdecs(elite_idx, :);
    end
    
    % 2. Select top 3 solutions by fitness
    [~, sorted_idx] = sort(popfits);
    top3 = popdecs(sorted_idx(1:3), :);
    
    % 3. Compute direction vectors
    d_elite = bsxfun(@minus, x_elite, popdecs);
    
    weights = [0.5, 0.3, 0.2];
    d_fit = zeros(NP, D);
    for k = 1:3
        d_fit = d_fit + weights(k) * bsxfun(@minus, top3(k,:), popdecs);
    end
    
    % 4. Adaptive scaling factors based on constraint violation
    max_cons = max(abs(cons));
    F = 0.2 + 0.6./(1 + exp(-5*abs(cons)/max_cons));
    F = F(:, ones(1,D));
    
    % 5. Mutation with Gaussian noise for diversity
    mutants = popdecs + F .* (0.7*d_elite + 0.3*d_fit) + 0.1*randn(NP, D);
    
    % 6. Rank-based crossover
    [~, rank_idx] = sort(popfits);
    ranks = zeros(NP, 1);
    ranks(rank_idx) = 1:NP;
    CR = 0.5 + 0.4*(1 - ranks/NP);
    CR = CR(:, ones(1,D));
    
    % Binomial crossover with j_rand protection
    mask = rand(NP, D) < CR;
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % 7. Boundary handling with reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = 2*lb(lb_mask) - offspring(lb_mask);
    offspring(ub_mask) = 2*ub(ub_mask) - offspring(ub_mask);
    
    % Final bounds check
    offspring = min(max(offspring, lb), ub);
end
% MATLAB Code
function [offspring] = updateFunc1295(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Select elite solution
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        temp = find(feasible);
        x_elite = popdecs(temp(elite_idx), :);
    else
        [~, elite_idx] = min(cons);
        x_elite = popdecs(elite_idx, :);
    end
    
    % 2. Select top 3 solutions
    [~, sorted_idx] = sort(popfits);
    top3 = popdecs(sorted_idx(1:3), :);
    weights = [0.5; 0.3; 0.2];
    
    % 3. Compute direction vectors
    d_elite = bsxfun(@minus, x_elite, popdecs);
    d_top = zeros(NP, D);
    for k = 1:3
        d_top = d_top + weights(k) * bsxfun(@minus, top3(k,:), popdecs);
    end
    
    % 4. Adaptive scaling factors
    min_cons = min(cons);
    max_cons = max(cons);
    if max_cons > min_cons
        norm_cons = (cons - min_cons) / (max_cons - min_cons);
    else
        norm_cons = zeros(size(cons));
    end
    F = 0.5 + 0.3 * norm_cons;
    F = F(:, ones(1,D));
    
    % 5. Mutation with Gaussian noise
    mutants = popdecs + F .* (0.6*d_elite + 0.4*d_top) + 0.1*randn(NP, D);
    
    % 6. Rank-based crossover
    if any(feasible)
        [~, rank_idx] = sort(popfits);
    else
        [~, rank_idx] = sort(cons);
    end
    ranks = zeros(NP, 1);
    ranks(rank_idx) = 1:NP;
    CR = 0.8 - 0.4*(ranks/NP);
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
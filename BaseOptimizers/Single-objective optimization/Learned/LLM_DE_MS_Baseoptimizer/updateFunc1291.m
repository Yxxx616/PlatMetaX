% MATLAB Code
function [offspring] = updateFunc1291(popdecs, popfits, cons)
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
    
    % 2. Select top 3 solutions by fitness
    [~, sorted_idx] = sort(popfits);
    top3 = popdecs(sorted_idx(1:3), :);
    
    % 3. Compute direction vectors
    d_elite = x_elite - popdecs;
    
    weights = [0.5, 0.3, 0.2];
    d_fit = zeros(NP, D);
    for k = 1:3
        d_fit = d_fit + weights(k) * (top3(k,:) - popdecs);
    end
    
    % 4. Adaptive scaling factors
    max_cons = max(abs(cons));
    F = 0.3 + 0.5./(1 + exp(-10*abs(cons)/max_cons));
    F = F(:, ones(1,D));
    
    % 5. Mutation
    mutants = popdecs + F .* (0.6*d_elite + 0.4*d_fit);
    
    % 6. Rank-based crossover
    [~, rank_idx] = sort(popfits);
    ranks = zeros(NP, 1);
    ranks(rank_idx) = 1:NP;
    CR = 0.6 + 0.3*(1 - ranks/NP);
    CR = CR(:, ones(1,D));
    
    % Binomial crossover
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
    
    % Ensure final bounds
    offspring = min(max(offspring, lb), ub);
end
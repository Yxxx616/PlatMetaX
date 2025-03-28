% MATLAB Code
function [offspring] = updateFunc1303(popdecs, popfits, cons)
    [NP, D] = size(popdecs);
    lb = -100 * ones(1, D);
    ub = 100 * ones(1, D);
    
    % 1. Select elite solution with adaptive strategy
    feasible = cons <= 0;
    if any(feasible)
        [~, elite_idx] = min(popfits(feasible));
        temp = find(feasible);
        x_elite = popdecs(temp(elite_idx), :);
    else
        [~, best_fit_idx] = min(popfits);
        [~, best_cons_idx] = min(cons);
        x_elite = 0.7*popdecs(best_fit_idx,:) + 0.3*popdecs(best_cons_idx,:);
    end
    
    % 2. Compute direction vectors
    d_elite = bsxfun(@minus, x_elite, popdecs);
    
    % Diversity direction vectors (random pairs)
    r1 = randi(NP, NP, 1);
    r2 = randi(NP, NP, 1);
    while any(r1 == r2)
        r2(r1 == r2) = randi(NP, sum(r1 == r2), 1);
    end
    d_div = popdecs(r1,:) - popdecs(r2,:);
    
    % 3. Adaptive weights calculation
    min_fit = min(popfits);
    max_fit = max(popfits);
    min_cons = min(cons);
    max_cons = max(cons);
    
    alpha = (popfits - min_fit) / (max_fit - min_fit + eps);
    beta = (cons - min_cons) / (max_cons - min_cons + eps);
    
    % Sigmoid weighting
    w = 1 ./ (1 + exp(-5*(alpha - beta)));
    w = w(:, ones(1,D));
    
    % 4. Adaptive scaling factor
    F = 0.5 + 0.3 * rand(NP,1) .* (1 - w(:,1));
    F = min(max(F, 0.3), 0.9);
    F = F(:, ones(1,D));
    
    % Combined mutation
    mutants = popdecs + F .* (w.*d_elite + (1-w).*d_div);
    
    % 5. Rank-based crossover
    [~, rank_idx] = sort(popfits);
    rank = zeros(NP,1);
    rank(rank_idx) = (1:NP)';
    CR = 0.9 - 0.5*(rank/NP);
    CR = CR(:, ones(1,D));
    
    mask = rand(NP, D) < CR;
    j_rand = randi(D, NP, 1);
    mask = mask | bsxfun(@eq, (1:D), j_rand);
    
    offspring = popdecs;
    offspring(mask) = mutants(mask);
    
    % Boundary handling with reflection
    lb_mask = offspring < lb;
    ub_mask = offspring > ub;
    offspring(lb_mask) = 2*lb(lb_mask) - offspring(lb_mask);
    offspring(ub_mask) = 2*ub(ub_mask) - offspring(ub_mask);
    
    % Final bounds check
    offspring = min(max(offspring, lb), ub);
end